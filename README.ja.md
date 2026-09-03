# krea2-checkpoint-lora-merger

[English](README.md) | **日本語**

`merge_krea2_turbo_lora.py` は、1つ以上の Krea 2 LoRA を Krea 2 Turbo チェックポイントに直接マージし、LoRA適用済みの単一モデルとして出力するツールです。ComfyUI の量子化チェックポイント (int8 / FP8) にも対応しています。

マージ結果は推論時に LoRA をロードせずそのまま使えます。

## インストール

Python 3.10 以上と [uv](https://docs.astral.sh/uv/) を使用します。

```bash
# uv が未インストールの場合
curl -LsSf https://astral.sh/uv/install.sh | sh

# 仮想環境の作成 (.venv/)
uv venv

# 依存関係のインストール
uv pip install -r requirements.txt
```

CUDA 版 torch が必要な場合は `uv pip install` に `--index-url` を付けます (例: CUDA 13.0):

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu130
uv pip install safetensors
```

### 実行

```bash
uv run python merge_krea2_turbo_lora.py --help
```

## 対応フォーマット

### ベースチェックポイント

| 形式 | 判別方法 | 備考 |
|---|---|---|
| BF16 プレーン | `*.comfy_quant` キーなし | そのまま BF16 でマージ |
| int8 + convrot | `comfy_quant` = `{"format": "int8_tensorwise", "convrot": true, ...}` | Hadamard 回転を含めて正しくデ量子化 |
| FP8 | `comfy_quant` = `{"format": "float8_e4m3fn" / "float8_e5m2"}` | per-tensor スケール。uint8 旧形式ストレージにも対応 |

- キー先頭の `model.diffusion_model.` / `diffusion_model.` / `model.transformer.` / `transformer.` プレフィクスは自動で剥がして LoRA と照合します(出力は元のプレフィクスを維持)。
- モジュールごとの `comfy_quant` JSON (`full_precision_matrix_mult` など) は出力時に保持します。
- `txtfusion.layerwise_blocks` / `txtfusion.refiner_blocks` など、ベースに存在するモジュールであれば LoRA 側にキーがあればマージ対象になります。

### LoRA

| 形式 | キー例 | 変換 |
|---|---|---|
| musubi-tuner 標準 | `lora_unet_blocks_0_attn_wq.lora_down.weight` | そのまま使用 |
| ai-toolkit / PEFT | `diffusion_model.blocks.0.attn.wq.lora_A.weight` | 自動で musubi 形式へ変換 |

- ai-toolkit 形式は alpha を保存しないため、alpha = rank (スケール1.0) を付与します。これは ComfyUI と同じ挙動で、ai-toolkit が lora_B 側にスケールを織り込んでいる前提です。
- `transformer.` / `diffusion_model.` 以外のキー (テキストエンコーダ等) はスキップされます。
- LoHa / LoKr には対応していません (エラーになります)。

## 使い方

```bash
python merge_krea2_turbo_lora.py \
    --base  <ベースチェックポイント.safetensors> \
    --lora  <LoRA1.safetensors> <LoRA2.safetensors> ... \
    --multiplier 0.7 0.6 ... \
    --output <出力先.safetensors>
```

`merge.sh` に上記を書いて実行するのが簡単です。

### オプション

| オプション | デフォルト | 説明 |
|---|---|---|
| `--base` | (必須) | ベースチェックポイント |
| `--lora` | (必須) | LoRA ファイル (複数指定可) |
| `--multiplier` | 全部 1.0 | LoRA ごとの強度。個数が足りない分は 1.0 扱い |
| `--output` | (必須) | 出力先。ベースと同じパスは拒否 |
| `--output-format` | `same` | `same` / `int8` / `bf16` (下記参照) |
| `--calc-device` | `auto` | 行列演算デバイス。`auto` は CUDA があれば使用 |
| `--force` | off | 出力先が既に存在しても上書き |

### 出力形式 (`--output-format`)

| 値 | 動作 | サイズ目安 | 品質 |
|---|---|---|---|
| `same` | 各モジュールをベースと同じ量子化形式で再量子化 | base と同程度 | 再量子化ノイズあり |
| `int8` | すべて int8_tensorwise + convrot (group 256) に変換 | 約 1/2 | 再量子化ノイズあり |
| `bf16` | 非量子化 BF16 として出力 (`comfy_quant` / `weight_scale` は削除) | 約 2 倍 | ノイズなし |

- ベースが非量子化の場合、`same` は `bf16` 相当になります。
- ComfyUI はモジュールごとに `comfy_quant` の有無を判定するため、量子化 / 非量子化が混在したチェックポイントも問題なくロードできます。

## 動作原理

量子化モジュールは以下のパイプラインで処理します (FP32 はすべて float32 演算)。

1. **デ量子化**
   - int8 + convrot: `W = rotate(q.float() × weight_scale, H, group_size)`
     - `H` は正規化直交 Hadamard 行列 (group_size = 256 なら Kronecker 冪 `h4⊗h4⊗h4⊗h4 / 16`)。H は対称かつ直交なので回転は自己逆変換になります。
   - FP8: `W = w.float() × weight_scale` (per-tensor スケール)
2. **LoRA マージ**: `W ← W + multiplier × (lora_up @ lora_down) × (alpha / rank)`
3. **再量子化**
   - int8 + convrot: `W` を回転 → 行ごとに `scale = absmax / 127` → 四捨五入して clamp(-128, 127)
   - FP8: `scale = absmax / 448` (e4m3) または `/ 57344` (e5m2) → clamp のうえ FP8 へキャスト

デ量子化・再量子化の実装は ComfyUI の `comfy_kitchen` (eager 実装) とビット一致することを検証済みです。スカラー係数は厳密には `Σ mult × (U @ D) × alpha/rank` の順で FP32 加算します。

非量子化モジュールは FP32 でマージした後 BF16 で保存します (LoRA 差分はビット一致を検証済み)。norm スケールやバイアス、`mod.lin` などの非線形重みは元の dtype のままコピーします。

## 品質に関する注意

- 量子化形式で出力する場合、マージ後の重みを再度量子化するため、量子化誤差 (ノイズ) が乗ります。これはベースチェックポイント自体が持っている誤差と同種・同程度のものです。
- FP8 (e4m3) は約 3 bit 仮数のため誤差が大きめです。弱い LoRA (差分の最大値が重み標準偏差の数%程度) だと、LoRA の効果が再量子化ノイズに埋もれ気味になることがあります。確実に効果を残したい場合は `--output-format bf16` を使ってください。
- int8 (行単位スケール) は FP8 より高精度で、サイズも同じく約 1/2 です。

## 制限事項

- LoHa / LoKr、および `nvfp4` / `mxfp8` / `convrot_w4a4` / `asym_w4a8_int8` 形式の量子化チェックポイントには非対応 (明確にエラーになります)。
- Linear (2D) 重みのみ想定しています。Conv 層への LoRA には対応していません (Krea2 の通常の LoRA はすべて Linear です)。
- 1つのモジュールに複数 LoRA が同じターゲットを持つ場合、指定した順に加算します。
