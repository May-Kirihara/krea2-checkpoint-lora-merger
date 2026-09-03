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
uv run python merge_krea2_turbo_lora.py --help
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
python merge_krea2_turbo_lora.py \
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
| `--output-format` | `same` | `same` / `int8` / `bf16` (see below) |
| `--calc-device` | `auto` | Device for matrix math. `auto` uses CUDA when available |
| `--force` | off | Overwrite an existing output file |

### Output formats (`--output-format`)

| Value | Behavior | Approx. size | Quality |
|---|---|---|---|
| `same` | Re-quantizes each module in the same format as the base | same as base | requantization noise |
| `int8` | Converts everything to int8_tensorwise + convrot (group 256) | ~1/2 | requantization noise |
| `bf16` | Outputs unquantized BF16 (`comfy_quant` / `weight_scale` are removed) | ~2x | no noise |

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
   - int8 + convrot: rotate `W` → per row `scale = absmax / 127` → round and clamp(-128, 127)
   - FP8: `scale = absmax / 448` (e4m3) or `/ 57344` (e5m2) → clamp, then cast to FP8

The dequantization/requantization implementations are verified to be bit-exact against ComfyUI's `comfy_kitchen` (eager implementation). Scalar coefficients are strictly accumulated in FP32 in the order `Σ mult × (U @ D) × alpha/rank`.

Unquantized modules are merged in FP32 and saved as BF16 (the LoRA deltas are verified to be bit-exact). Norm scales, biases, and non-linear weights such as `mod.lin` are copied with their original dtype.

## Quality Notes

- When outputting in a quantized format, the merged weights are quantized again, which adds quantization error (noise) — of the same kind and magnitude as the base checkpoint itself already carries.
- FP8 (e4m3) has a ~3-bit mantissa and thus larger errors. With weak LoRAs (max delta on the order of a few percent of the weight standard deviation), the LoRA effect can get buried in requantization noise. Use `--output-format bf16` when you want to preserve the effect reliably.
- int8 (per-row scaling) is more accurate than FP8 at the same ~1/2 size.

## Limitations

- LoHa / LoKr, as well as `nvfp4` / `mxfp8` / `convrot_w4a4` / `asym_w4a8_int8` quantized checkpoints are not supported (an explicit error is raised).
- Only Linear (2D) weights are assumed. LoRAs targeting Conv layers are not supported (normal Krea2 LoRAs are all Linear).
- When multiple LoRAs target the same module, they are added in the order specified.
