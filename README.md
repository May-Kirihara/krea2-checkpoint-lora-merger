# krea2-checkpoint-lora-merger

**English** | [日本語](README.ja.md)

`merge_krea2_turbo_lora.py` merges one or more Krea 2 LoRAs directly into a Krea 2 Turbo checkpoint and outputs a single standalone model with the LoRAs baked in. Quantized ComfyUI checkpoints (int8 / FP8) are supported.

The merged model can be used as-is at inference time, without loading any LoRA.

## Installation

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment (.venv/)
uv venv

# Install dependencies
uv pip install -r requirements.txt
```

If you need the CUDA build of torch, add `--index-url` to `uv pip install` (example: CUDA 13.0):

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu130
uv pip install safetensors
```

### Running

```bash
uv run python merge_krea2_lora.py --help
```

## Supported Formats

### Base checkpoint

| Format | Detection | Notes |
|---|---|---|
| BF16 plain | no `*.comfy_quant` keys | Merged as BF16 as-is |
| int8 + convrot | `comfy_quant` = `{"format": "int8_tensorwise", "convrot": true, ...}` | Dequantized correctly, including the Hadamard rotation |
| FP8 | `comfy_quant` = `{"format": "float8_e4m3fn" / "float8_e5m2"}` | Per-tensor scale. Legacy uint8 storage is also supported |

- Leading `model.diffusion_model.` / `diffusion_model.` / `model.transformer.` / `transformer.` key prefixes are automatically stripped when matching against LoRA keys (the output keeps the original prefixes).
- Per-module `comfy_quant` JSON (e.g. `full_precision_matrix_mult`) is preserved on output.
- Any module present in the base — such as `txtfusion.layerwise_blocks` / `txtfusion.refiner_blocks` — is merged if the LoRA has keys for it.

### LoRA

| Format | Example key | Conversion |
|---|---|---|
| musubi-tuner standard | `lora_unet_blocks_0_attn_wq.lora_down.weight` | Used as-is |
| ai-toolkit / PEFT | `diffusion_model.blocks.0.attn.wq.lora_A.weight` | Automatically converted to musubi format |

- ai-toolkit LoRAs do not store alpha, so alpha = rank (scale 1.0) is assigned. This matches ComfyUI's behavior and assumes ai-toolkit bakes the scale into the lora_B weights.
- Keys outside `transformer.` / `diffusion_model.` (text encoders, etc.) are skipped.
- LoHa / LoKr are not supported (an error is raised).

## Usage

```bash
python merge_krea2_lora.py \
    --base  <base_checkpoint.safetensors> \
    --lora  <LoRA1.safetensors> <LoRA2.safetensors> ... \
    --multiplier 0.7 0.6 ... \
    --output <output.safetensors>
```

The easiest way is to write the command into `merge.sh` and run it.

### Options

| Option | Default | Description |
|---|---|---|
| `--base` | (required) | Base checkpoint |
| `--lora` | (required) | LoRA file(s); multiple can be specified |
| `--multiplier` | all 1.0 | Per-LoRA strength. Missing entries are treated as 1.0 |
| `--output` | (required) | Output path. Identical to the base path is rejected |
| `--output-format` | `same` | `same` / `int8` / `bf16` / `hybrid` (see below) |
| `--scale-search` | `mse` | Scale selection when re-quantizing. `mse`: grid-search a clipping factor per row (int8) / per tensor (fp8) over `absmax*[0.5..1.0]` and keep the scale with the lowest reconstruction MSE (absmax = 1.0 is in the grid, so it is never worse). `off`: plain absmax |
| `--hybrid-snr` | `2.0` | `hybrid` only. Promote a module to BF16 when `‖LoRA delta‖ / ‖re-quantization error‖` falls below this |
| `--calc-device` | `auto` | Device for matrix math. `auto` uses CUDA when available |
| `--force` | off | Overwrite an existing output file |

### Output formats (`--output-format`)

| Value | Behavior | Approx. size | Quality |
|---|---|---|---|
| `same` | Re-quantizes LoRA-touched modules in the same format as the base; untouched modules are copied unchanged | same as base | requantization noise (reduced by the MSE scale search) |
| `int8` | Converts everything to int8_tensorwise + convrot; untouched int8 modules are copied | ~1/2 | requantization noise (reduced by the MSE scale search) |
| `bf16` | Outputs unquantized BF16 (`comfy_quant` / `weight_scale` are removed) | ~2x | no noise |
| `hybrid` | Per-module SNR test on LoRA-touched modules: promote to BF16 only where the delta would be buried in requantization noise; keep the original quantized format elsewhere. Untouched modules are copied unchanged | between base and bf16 (depends on LoRA strength) | promoted modules are lossless |

- If the base is unquantized, `same` behaves like `bf16`.
- ComfyUI checks `comfy_quant` per module, so checkpoints mixing quantized / unquantized modules load fine.

## How It Works

Quantized modules go through the following pipeline (all FP32 math is float32).

1. **Dequantization**
   - int8 + convrot: `W = rotate(q.float() × weight_scale, H, group_size)`
     - `H` is a normalized orthogonal Hadamard matrix (for group_size = 256, the Kronecker power `h4⊗h4⊗h4⊗h4 / 16`). H is symmetric and orthogonal, so the rotation is its own inverse.
   - FP8: `W = w.float() × weight_scale` (per-tensor scale)
2. **LoRA merge**: `W ← W + multiplier × (lora_up @ lora_down) × (alpha / rank)`
3. **Re-quantization**
   - int8 + convrot: rotate `W` → choose a per-row scale → round and clamp(-128, 127)
   - FP8: choose a per-tensor scale → clamp, then cast to FP8
   - By default the scale is chosen by MSE search (`--scale-search mse`): clipping factors α from 1.00 down to 0.50 (step 0.02) are tried and the scale with the lowest reconstruction MSE is kept per row (int8) / per tensor (fp8). Since α = 1.0 (absmax) is in the grid, the result is never worse than plain absmax.

Quantized tensors of modules the LoRA does not touch (`weight` / `weight_scale` / `comfy_quant`) are copied byte-for-byte with zero extra noise.

`hybrid` decision: each LoRA-touched module is tentatively re-quantized; if `‖LoRA delta‖ / ‖re-quantization error‖` falls below `--hybrid-snr` (default 2.0) the module is promoted to BF16 instead — trading size for fidelity only where the delta would be buried in noise.

The dequantization/requantization implementations are verified to be bit-exact against ComfyUI's `comfy_kitchen` (eager implementation). Scalar coefficients are strictly accumulated in FP32 in the order `Σ mult × (U @ D) × alpha/rank`.

Unquantized modules are merged in FP32 and saved as BF16 (the LoRA deltas are verified to be bit-exact). Norm scales, biases, and non-linear weights such as `mod.lin` are copied with their original dtype.

## Quality Notes

- When outputting in a quantized format, the merged weights are quantized again, which adds quantization error (noise) — of the same kind and magnitude as the base checkpoint itself already carries.
- FP8 (e4m3) has a ~3-bit mantissa and thus larger errors. With weak LoRAs (max delta on the order of a few percent of the weight standard deviation), the LoRA effect can get buried in requantization noise.
- int8 (per-row scaling) is more accurate than FP8 at the same ~1/2 size. Note, however, that with weak LoRAs or low multipliers the delta can be comparable to the requantization noise (measured example: int8 base + style LoRA ×0.7 gave a per-module SNR median of ≈ 1.3).
- To match runtime LoRA application (ForgeNEO etc.) exactly, use `--output-format bf16`. `hybrid` matches exactly on every module it promotes; weak LoRAs promote most modules (approaching bf16 size/quality), strong LoRAs keep the quantized size while rescuing only the degraded modules. Lower `--hybrid-snr` (e.g. 1.0 keeps a module quantized as long as the delta is at least at the noise floor) to prioritize size.

## Limitations

- LoHa / LoKr, as well as `nvfp4` / `mxfp8` / `convrot_w4a4` / `asym_w4a8_int8` quantized checkpoints are not supported (an explicit error is raised).
- Only Linear (2D) weights are assumed. LoRAs targeting Conv layers are not supported (normal Krea2 LoRAs are all Linear).
- When multiple LoRAs target the same module, they are added in the order specified.
