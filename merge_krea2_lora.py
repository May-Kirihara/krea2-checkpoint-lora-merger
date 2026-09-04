#!/usr/bin/env python3
"""
Merge one or more Krea 2 LoRAs into a Krea 2 Turbo checkpoint.

Supports:
  - base: Krea 2 Turbo BF16 checkpoint, or ComfyUI quantized checkpoint
    (int8_tensorwise + convrot, or float8_e4m3fn / float8_e5m2, i.e. *.comfy_quant keys)
  - base key prefixes "model.diffusion_model." / "diffusion_model." are handled
  - LoRA: musubi-tuner format (lora_unet_...lora_down/lora_up) or
    ai-toolkit / PEFT format (diffusion_model....lora_A/lora_B), converted automatically
  - output: re-quantized to the same format as the base (default), int8+convrot
    (--output-format int8), plain BF16 (--output-format bf16), or a hybrid that
    keeps quantization only where the LoRA delta survives re-quantization
    (--output-format hybrid, promoting weak-SNR modules to BF16)
  - re-quantization uses a per-row (int8) / per-tensor (fp8) MSE-optimal scale
    grid search by default (--scale-search mse); alpha=1.0 (plain absmax) is
    included in the grid, so results are never worse than min-max quantization

The output is usable as a standalone DiT checkpoint without loading the LoRA again.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge Krea2 LoRA(s) into a Krea2 Turbo checkpoint (BF16 or ComfyUI int8+convrot)."
    )
    p.add_argument("--base", required=True, help="Krea2 Turbo base checkpoint (.safetensors).")
    p.add_argument("--lora", required=True, nargs="+", help="One or more Krea2 LoRA .safetensors files.")
    p.add_argument("--output", required=True, help="Output merged .safetensors path.")
    p.add_argument(
        "--multiplier",
        type=float,
        nargs="*",
        default=None,
        help="Multiplier per LoRA. Missing values default to 1.0.",
    )
    p.add_argument(
        "--output-format",
        choices=["same", "int8", "bf16", "hybrid"],
        default="same",
        help=(
            "Output format. 'same': dequantize -> merge -> re-quantize each module to its "
            "original base format (int8+convrot or fp8; default). 'int8': re-quantize everything "
            "to int8_tensorwise + convrot. 'bf16': plain BF16 checkpoint (no quant keys). "
            "'hybrid': keep each LoRA-touched module quantized only when the LoRA delta is "
            "clearly larger than the re-quantization error (see --hybrid-snr); promote it to "
            "BF16 otherwise. Untouched modules are always copied unchanged."
        ),
    )
    p.add_argument(
        "--scale-search",
        choices=["mse", "off"],
        default="mse",
        help=(
            "Scale selection when re-quantizing. 'mse': grid-search a clipping factor per row "
            "(int8) / per tensor (fp8) over absmax*[0.5..1.0] and keep the scale with the "
            "lowest reconstruction MSE (default). 'off': plain absmax scaling."
        ),
    )
    p.add_argument(
        "--hybrid-snr",
        type=float,
        default=2.0,
        metavar="RATIO",
        help=(
            "hybrid output only: promote a LoRA-touched module to BF16 when "
            "||LoRA delta|| / ||re-quantization error|| falls below this ratio (default 2.0)."
        ),
    )
    p.add_argument(
        "--calc-device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for matrix math. 'auto' uses CUDA when available.",
    )
    p.add_argument("--force", action="store_true", help="Overwrite output if it already exists.")
    return p.parse_args()


def normalize_multipliers(values: List[float] | None, count: int) -> List[float]:
    if values is None:
        return [1.0] * count
    values = list(values)
    if len(values) < count:
        values.extend([1.0] * (count - len(values)))
    return values[:count]


def get_metadata(path: str) -> Dict[str, str]:
    with safe_open(path, framework="pt", device="cpu") as f:
        md = f.metadata()
    return dict(md) if md else {}


# ----------------------------------------------------------------------------
# ComfyUI int8_tensorwise + convrot helpers
# (bit-exact re-implementation of comfy_kitchen.backends.eager.quantization /
#  comfy_kitchen.tensor.int8_utils; H is symmetric and orthogonal)
# ----------------------------------------------------------------------------

_H4 = [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]]
_H_CACHE: Dict[int, torch.Tensor] = {}


def build_hadamard(size: int) -> torch.Tensor:
    if size in _H_CACHE:
        return _H_CACHE[size]
    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Regular Hadamard size must be a power of 4, got {size}")
    h4 = torch.tensor(_H4, dtype=torch.float32)
    h = h4
    cur = 4
    while cur < size:
        h = torch.kron(h, h4)
        cur *= 4
    h = h / (size**0.5)
    _H_CACHE[size] = h
    return h


def rotate_weight(w: torch.Tensor, h: torch.Tensor, group_size: int) -> torch.Tensor:
    out_f, in_f = w.shape
    n_groups = in_f // group_size
    return torch.matmul(w.reshape(out_f, n_groups, group_size), h.T.to(w.dtype)).reshape(out_f, in_f)


def dequantize_int8_convrot(q: torch.Tensor, scale: torch.Tensor, group_size: int, h: torch.Tensor) -> torch.Tensor:
    return rotate_weight(q.to(torch.float32) * scale.to(torch.float32), h, group_size)


# clipping-factor candidates for the MSE scale search; 1.0 reproduces plain
# absmax quantization, so the search result is never worse than min-max
_MSE_ALPHAS: Tuple[float, ...] = tuple(round(1.0 - 0.02 * i, 2) for i in range(26))


def quantize_int8_convrot(
    w: torch.Tensor, group_size: int, h: torch.Tensor, scale_search: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    w_rot = rotate_weight(w, h, group_size)
    abs_max = w_rot.abs().amax(dim=-1, keepdim=True)
    if not scale_search:
        scale = (abs_max / 127.0).clamp(min=1e-30)
        q = torch.round(w_rot / scale).clamp_(-128, 127).to(torch.int8)
        return q, scale
    best_q: torch.Tensor | None = None
    best_scale: torch.Tensor | None = None
    best_err: torch.Tensor | None = None
    for alpha in _MSE_ALPHAS:
        scale = (abs_max * alpha / 127.0).clamp(min=1e-30)
        q = torch.round(w_rot / scale).clamp(-128, 127)
        err = (q * scale - w_rot).pow(2).sum(dim=-1, keepdim=True)
        if best_err is None:
            best_q, best_scale, best_err = q, scale, err
            continue
        better = err < best_err
        best_q = torch.where(better, q, best_q)
        best_scale = torch.where(better, scale, best_scale)
        best_err = torch.where(better, err, best_err)
    assert best_q is not None and best_scale is not None
    return best_q.to(torch.int8), best_scale


def parse_comfy_quant(t: torch.Tensor) -> dict:
    return json.loads(t.numpy().tobytes())


def comfy_quant_tensor(conf: dict) -> torch.Tensor:
    return torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8)


# ----------------------------------------------------------------------------
# FP8 helpers (bit-exact re-implementation of comfy_kitchen eager ops)
# ----------------------------------------------------------------------------

FP8_TYPES = {
    "float8_e4m3fn": (torch.float8_e4m3fn, 448.0),
    "float8_e5m2": (torch.float8_e5m2, 57344.0),
}


def dequantize_fp8(w: torch.Tensor, scale: torch.Tensor, fmt: str) -> torch.Tensor:
    dtype, _ = FP8_TYPES[fmt]
    if w.dtype == torch.uint8:  # legacy fp8-on-disk-as-uint8 storage
        w = w.view(dtype)
    return w.to(torch.float32) * scale.to(torch.float32)


def quantize_fp8(w: torch.Tensor, fmt: str, scale_search: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    dtype, fmax = FP8_TYPES[fmt]
    absmax = w.abs().amax().to(torch.float32)
    if not scale_search:
        scale = (absmax / fmax).clamp(min=1e-30).reshape(())
        q = torch.clamp(w * (1.0 / scale).to(w.dtype), -fmax, fmax).to(dtype)
        return q, scale
    best_q = None
    best_scale = None
    best_err = None
    for alpha in _MSE_ALPHAS:
        scale = ((absmax * alpha) / fmax).clamp(min=1e-30).reshape(())
        q = torch.clamp(w * (1.0 / scale).to(w.dtype), -fmax, fmax).to(dtype)
        err = ((q.to(torch.float32) * scale) - w).pow(2).sum()
        if best_err is None or bool(err < best_err):
            best_q, best_scale, best_err = q, scale, err
    assert best_q is not None and best_scale is not None
    return best_q, best_scale


# ----------------------------------------------------------------------------
# Module prefix handling
# ----------------------------------------------------------------------------

_BASE_KEY_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "model.transformer.", "transformer.")


def strip_base_prefix(module: str) -> str:
    for p in _BASE_KEY_PREFIXES:
        if module.startswith(p):
            return module[len(p):]
    return module


# ----------------------------------------------------------------------------
# LoRA handling
# ----------------------------------------------------------------------------

def convert_aitoolkit_lora_keys(
    lora_sd: Dict[str, torch.Tensor],
    lora_path: str,
) -> Dict[str, torch.Tensor]:
    """
    Convert ai-toolkit / PEFT style LoRA keys to musubi-tuner format:

        diffusion_model.blocks.0.attn.wq.lora_A.weight
            -> lora_unet_blocks_0_attn_wq.lora_down.weight
        diffusion_model.blocks.0.attn.wq.lora_B.weight
            -> lora_unet_blocks_0_attn_wq.lora_up.weight

    ai-toolkit does not save alpha (its scaling is baked into lora_B), so alpha
    is set equal to rank, i.e. scale = 1.0. Same behavior as ComfyUI.
    """
    if not any(k.endswith((".lora_A.weight", ".lora_B.weight")) for k in lora_sd.keys()):
        return lora_sd  # already musubi format

    new_sd: Dict[str, torch.Tensor] = {}
    lora_dims: Dict[str, int] = {}
    skipped = 0
    for key, weight in lora_sd.items():
        if not key.startswith(("diffusion_model.", "transformer.")):
            skipped += 1  # e.g. text encoder modules
            continue
        body = key.split(".", 1)[1]
        new_key = ("lora_unet_" + body.replace(".", "_")).replace("_lora_A_", ".lora_down.").replace(
            "_lora_B_", ".lora_up."
        )
        new_sd[new_key] = weight
        lora_name = new_key.split(".", 1)[0]
        if "lora_down" in new_key and lora_name not in lora_dims:
            lora_dims[lora_name] = weight.shape[0]

    for lora_name, dim in lora_dims.items():
        new_sd[f"{lora_name}.alpha"] = torch.tensor(dim)

    print(f"      {Path(lora_path).name}: converted ai-toolkit/PEFT keys to musubi format")
    if skipped:
        print(f"      {Path(lora_path).name}: skipped {skipped} non-transformer keys")
    return new_sd


def build_lora_entries(
    base_keys: Set[str],
    lora_sd: Dict[str, torch.Tensor],
    lora_path: str,
) -> Dict[str, List[Tuple[torch.Tensor, torch.Tensor, float]]]:
    """
    Map base module name -> [(lora_down, lora_up, scale), ...] for one LoRA.
    musubi-tuner maps base key `blocks.0.attn.to_q.weight` to LoRA prefix
    `lora_unet_blocks_0_attn_to_q` + `.lora_down.weight` / `.lora_up.weight`.
    """
    lora_keys = set(lora_sd.keys())

    if any("hada_w1_a" in k for k in lora_keys):
        raise RuntimeError(f"{lora_path}: LoHa weights detected. This script is intended for standard LoRA.")
    if any("lokr_w1" in k for k in lora_keys):
        raise RuntimeError(f"{lora_path}: LoKr weights detected. This script is intended for standard LoRA.")

    entries: Dict[str, List[Tuple[torch.Tensor, torch.Tensor, float]]] = {}
    for base_key in base_keys:
        if not base_key.endswith(".weight"):
            continue
        module = strip_base_prefix(base_key.rsplit(".", 1)[0])
        prefix = "lora_unet_" + module.replace(".", "_")
        down = lora_sd.get(prefix + ".lora_down.weight")
        up = lora_sd.get(prefix + ".lora_up.weight")
        if down is None or up is None:
            continue
        rank = down.shape[0]
        alpha = lora_sd.get(prefix + ".alpha")
        alpha_val = float(alpha.item()) if alpha is not None else float(rank)
        entries.setdefault(module, []).append((down, up, alpha_val / rank))
    return entries


def main() -> None:
    args = parse_args()

    base = Path(args.base).expanduser().resolve()
    lora_paths = [Path(x).expanduser().resolve() for x in args.lora]
    output = Path(args.output).expanduser().resolve()

    if not base.is_file():
        raise SystemExit(f"Base checkpoint not found: {base}")
    for lp in lora_paths:
        if not lp.is_file():
            raise SystemExit(f"LoRA not found: {lp}")
    if output == base:
        raise SystemExit("Refusing to overwrite the base checkpoint. Choose a different --output.")
    if output.exists() and not args.force:
        raise SystemExit(f"Output already exists: {output}\nUse --force to overwrite it.")

    if args.calc_device == "auto":
        calc_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        calc_device = torch.device(args.calc_device)
    if calc_device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--calc-device cuda was requested, but CUDA is not available.")

    multipliers = normalize_multipliers(args.multiplier, len(lora_paths))

    print(f"[1/5] Base:   {base}")
    print(f"      Output: {output}  (format: {args.output_format})")
    print(f"      Calc device: {calc_device}")
    print("      LoRAs:")
    for lp, mult in zip(lora_paths, multipliers):
        print(f"        - {lp}  x {mult:g}")

    # ---- scan base structure -------------------------------------------------
    print("[2/5] Reading checkpoint/LoRA keys...")
    quant_conf: Dict[str, dict] = {}  # module -> comfy_quant json dict
    base_keys: Set[str] = set()
    with safe_open(str(base), framework="pt", device="cpu") as f:
        base_keys = set(f.keys())
        for key in f.keys():
            if key.endswith(".comfy_quant"):
                module = key[: -len(".comfy_quant")]
                conf = parse_comfy_quant(f.get_tensor(key))
                fmt = conf.get("format")
                if fmt != "int8_tensorwise" and fmt not in FP8_TYPES:
                    raise SystemExit(f"Unsupported comfy_quant format {fmt!r} at {module}")
                quant_conf[module] = conf

    n_quant = len(quant_conf)
    fmt_counts: Dict[str, int] = {}
    for conf in quant_conf.values():
        fmt_counts[conf["format"]] = fmt_counts.get(conf["format"], 0) + 1
    group_sizes = {int(c.get("convrot_groupsize", 256)) for c in quant_conf.values() if c["format"] == "int8_tensorwise"}
    group_size = max(group_sizes) if group_sizes else 256
    h_cache: Dict[int, torch.Tensor] = {}

    def get_h(size: int) -> torch.Tensor:
        if size not in h_cache:
            h_cache[size] = build_hadamard(size).to(calc_device)
        return h_cache[size]

    if n_quant:
        print(f"      Base: {n_quant} quantized modules ({', '.join(f'{v}x {k}' for k, v in fmt_counts.items())})")
        out_desc = {
            "same": "re-quantizing touched modules to their original formats",
            "int8": "re-quantizing everything to int8+convrot",
            "bf16": "saving as BF16",
            "hybrid": "keeping quantization only where the LoRA survives it, BF16 otherwise",
        }[args.output_format]
        print(f"      Dequantizing to FP32, merging LoRA, then {out_desc}")
        extra = f"      Scale search: {args.scale_search}"
        if args.output_format == "hybrid":
            extra += f" (hybrid SNR threshold {args.hybrid_snr:g})"
        print(extra)

    # ---- load LoRAs -----------------------------------------------------------
    all_lora_entries: List[Dict[str, List[Tuple[torch.Tensor, torch.Tensor, float]]]] = []
    for lp in lora_paths:
        with safe_open(str(lp), framework="pt", device="cpu") as f:
            sd = {k: f.get_tensor(k) for k in f.keys()}
        sd = convert_aitoolkit_lora_keys(sd, str(lp))
        entries = build_lora_entries(base_keys, sd, str(lp))
        matched = sum(len(v) for v in entries.values())
        if matched == 0:
            raise RuntimeError(
                f"{lp}: no LoRA modules matched the Krea2 base checkpoint.\n"
                "This usually means the LoRA is not a Krea2 LoRA, or its key format is unknown."
            )
        print(f"      {lp.name}: {matched} matched LoRA modules")
        all_lora_entries.append(entries)

    # ---- merge -----------------------------------------------------------------
    print("[3/5] Merging LoRA into Krea2 Turbo weights...")
    merged_sd: Dict[str, torch.Tensor] = {}
    n_requant = n_plain = n_copied = n_promoted = 0

    def apply_loras(w: torch.Tensor, module: str) -> torch.Tensor:
        for entries, mult in zip(all_lora_entries, multipliers):
            for down, up, lscale in entries.get(module, []):
                delta = (up.to(calc_device, torch.float32) @ down.to(calc_device, torch.float32)) * lscale
                w = w + mult * delta
        return w

    def module_touched(full_module: str) -> bool:
        stripped = strip_base_prefix(full_module)
        return any(entries.get(stripped) for entries in all_lora_entries)

    def hybrid_keep_quantized(w_merged: torch.Tensor, w_base: torch.Tensor, recon: torch.Tensor) -> bool:
        """Keep the module quantized when the LoRA delta clearly survives re-quantization."""
        delta = float((w_merged - w_base).norm())
        err = float((recon - w_merged).norm())
        if delta == 0.0 or err == 0.0:
            return True
        return (delta / err) >= args.hybrid_snr

    with safe_open(str(base), framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.endswith(".comfy_quant") or key.endswith(".weight_scale"):
                continue  # regenerated below (quantized output) or dropped (bf16)

            module = key.rsplit(".", 1)[0] if key.endswith(".weight") else None
            tensor = f.get_tensor(key)

            if module is not None and module in quant_conf:
                conf = quant_conf[module]
                in_fmt = conf["format"]
                in_group = int(conf.get("convrot_groupsize", 256)) if in_fmt == "int8_tensorwise" else group_size
                touched = module_touched(module)

                # Resolve the per-module output format.
                if args.output_format == "bf16":
                    out_fmt = "bf16"
                elif args.output_format == "int8":
                    if not touched and in_fmt == "int8_tensorwise" and in_group == group_size:
                        out_fmt = "copy"
                    else:
                        out_fmt = "int8_tensorwise"
                else:  # same / hybrid
                    out_fmt = "copy" if not touched else in_fmt

                if out_fmt == "copy":
                    # Untouched module: copy the base tensors verbatim (zero extra noise).
                    merged_sd[key] = tensor
                    merged_sd[module + ".weight_scale"] = f.get_tensor(module + ".weight_scale")
                    merged_sd[module + ".comfy_quant"] = f.get_tensor(module + ".comfy_quant")
                    n_copied += 1
                    continue

                out_group = group_size if args.output_format == "int8" else in_group
                mse = args.scale_search == "mse"
                scale = f.get_tensor(module + ".weight_scale")
                w = tensor.to(calc_device)
                if in_fmt == "int8_tensorwise":
                    w = dequantize_int8_convrot(w, scale.to(calc_device), in_group, get_h(in_group))
                else:
                    w = dequantize_fp8(w, scale.to(calc_device), in_fmt)
                w_base = w
                w = apply_loras(w, strip_base_prefix(module))

                if out_fmt == "bf16":
                    merged_sd[key] = w.to("cpu", torch.bfloat16).contiguous()
                    n_plain += 1
                elif out_fmt == "int8_tensorwise":
                    q, new_scale = quantize_int8_convrot(w, out_group, get_h(out_group), mse)
                    if args.output_format == "hybrid":
                        recon = dequantize_int8_convrot(q, new_scale, out_group, get_h(out_group))
                        if not hybrid_keep_quantized(w, w_base, recon):
                            merged_sd[key] = w.to("cpu", torch.bfloat16).contiguous()
                            n_promoted += 1
                            del w, w_base
                            continue
                    merged_sd[key] = q.to("cpu").contiguous()
                    merged_sd[module + ".weight_scale"] = new_scale.to("cpu", torch.float32).contiguous()
                    merged_sd[module + ".comfy_quant"] = comfy_quant_tensor(
                        {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": out_group}
                    )
                    n_requant += 1
                else:  # fp8 out (same / hybrid keep the module's original fp8 format)
                    q, new_scale = quantize_fp8(w, out_fmt, mse)
                    if args.output_format == "hybrid":
                        recon = dequantize_fp8(q, new_scale, out_fmt)
                        if not hybrid_keep_quantized(w, w_base, recon):
                            merged_sd[key] = w.to("cpu", torch.bfloat16).contiguous()
                            n_promoted += 1
                            del w, w_base
                            continue
                    merged_sd[key] = q.to("cpu").contiguous()
                    merged_sd[module + ".weight_scale"] = new_scale.to("cpu", torch.float32).contiguous()
                    merged_sd[module + ".comfy_quant"] = comfy_quant_tensor(dict(conf))
                    n_requant += 1
                del w, w_base
            elif module is not None:
                if not module_touched(module):
                    merged_sd[key] = tensor
                    n_copied += 1
                else:
                    w = apply_loras(tensor.to(calc_device, torch.float32), strip_base_prefix(module))
                    merged_sd[key] = w.to("cpu", torch.bfloat16).contiguous()
                    n_plain += 1
                    del w
            else:
                merged_sd[key] = tensor  # norm scales, biases, mod.lin, ... kept as-is

    if calc_device.type == "cuda":
        torch.cuda.synchronize()

    parts = [f"{n_requant} modules merged + re-quantized"]
    if n_promoted:
        parts.append(f"{n_promoted} modules promoted to BF16 (hybrid)")
    parts.append(f"{n_plain} modules merged in BF16")
    if n_copied:
        parts.append(f"{n_copied} modules copied unchanged (no LoRA)")
    print("      " + ", ".join(parts) + ".")

    # ---- save --------------------------------------------------------------------
    print("[4/5] Saving merged checkpoint...")
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")

    metadata = get_metadata(str(base))
    metadata.update(
        {
            "krea2_lora_merged": "true",
            "krea2_merge_base": base.name,
            "krea2_merge_loras": ";".join(lp.name for lp in lora_paths),
            "krea2_merge_multipliers": ";".join(f"{x:g}" for x in multipliers),
            "krea2_merge_output_format": args.output_format,
            "krea2_merge_scale_search": args.scale_search,
            "krea2_merge_hybrid_snr": f"{args.hybrid_snr:g}",
            "krea2_merge_tool": "merge_krea2_turbo_lora.py",
        }
    )

    try:
        save_file(merged_sd, str(tmp), metadata=metadata)
        os.replace(tmp, output)
    finally:
        if tmp.exists():
            tmp.unlink()

    size_gib = output.stat().st_size / (1024**3)
    print("[5/5] Done.")
    print(f"      Saved: {output}")
    print(f"      Size:  {size_gib:.2f} GiB")
    print()
    print("Use this file as the Krea2 Turbo DiT checkpoint WITHOUT loading the merged LoRA again.")


if __name__ == "__main__":
    main()
