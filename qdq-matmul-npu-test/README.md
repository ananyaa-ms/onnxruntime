# QDQ NPU Compatibility

Generate configurable QDQ MatMul and normalization models and run them on CPU or a Windows ML NPU execution provider.

## Set up

Create the model-generation environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
```

Create the Windows ML runner environment:

```powershell
python -m venv .venv-winml
.\.venv-winml\Scripts\python.exe -m pip install --upgrade pip
.\.venv-winml\Scripts\python.exe -m pip install --no-deps -r .\requirements-winml.txt
```

## Generate the unit model

`generate_qdq_matmul_model.py` creates a float32-input/output model with asymmetric uint16 activation QDQ, asymmetric uint8 constant-weight DQ, and MatMul. Weight quantization can be per-tensor, per-channel, or blockwise.

| QDQ profile | Standard opset | Q/DQ domain | Q/DQ opset |
|---|---:|---|---:|
| `onnx` | 21 | Default ONNX | 21 |
| `microsoft` | 17 | `com.microsoft` | 1 |

Select the profile with `--qdq-profile onnx` or `--qdq-profile microsoft`. Per-tensor and per-channel weights support both profiles; blockwise weights require the `onnx` profile because `com.microsoft::DequantizeLinear` has no `block_size` attribute.

Add `--add-vitisai-metadata` to write the CLIP model identity and Vitis AI quantization properties expected for uint16 activations, uint8 weights, and `OnnxStaticQuantization`.

```powershell
.\.venv\Scripts\python.exe .\generate_qdq_matmul_model.py `
    --weight-quantization blockwise `
    --block-size 32
```

The default input shape is `[1, 2520, 768]`, the default weight shape is `[768, 768]`, and the default output is `unit-models\clip_visual_dq_matmul_q.onnx`. Run `.\.venv\Scripts\python.exe .\generate_qdq_matmul_model.py -h` for all options.

## Generate normalization unit models

`generate_qdq_normalization_model.py` generates three activation-quantized patterns. Inputs and outputs are float32 so the runners can supply and compare them; uint16 Q/DQ pairs surround the operator boundaries, while constant normalization scales use uint8 DQ. The `add-lpnorm-mul` pattern takes same-shaped `input`, `addend`, and `multiplier` activation inputs.

| Pattern | Core graph | Default shape | Default output |
|---|---|---|---|
| `rmsnorm` | DQ -> `RMSNormalization` -> Q | `[1, 2520, 768]` | `unit-models\gemma_dq_rmsnorm_q.onnx` |
| `sslrn` | DQ inputs -> `com.microsoft::SkipSimplifiedLayerNormalization` -> Q | `[1, 2520, 768]` | `unit-models\gemma_dq_sslrn_q.onnx` |
| `add-lpnorm-mul` | QDQ `Add` -> QDQ `LpNormalization` -> QDQ `Mul` | `[1, 512]` (CLIP image) | `unit-models\clip_qdq_add_lpnorm_mul.onnx` |

Generate the models:

```powershell
.\.venv\Scripts\python.exe .\generate_qdq_normalization_model.py --pattern rmsnorm
.\.venv\Scripts\python.exe .\generate_qdq_normalization_model.py --pattern sslrn
.\.venv\Scripts\python.exe .\generate_qdq_normalization_model.py --pattern add-lpnorm-mul
```

The CLIP image LpNorm site is the default, with shape `[1, 512]`. Use `--clip-lpnorm-site text` for the original text site with shape `[10, 512]`:

```powershell
.\.venv\Scripts\python.exe .\generate_qdq_normalization_model.py `
    --pattern add-lpnorm-mul `
    --clip-lpnorm-site text
```

Both sites use their original input and output uint16 quantization parameters. Use `--input-shape DIM [DIM ...]` to test another concrete shape.

`--qdq-profile onnx` uses default-domain Q/DQ at opset 23. To match the original CLIP model's `com.microsoft` Q/DQ domain and opset, run:

```powershell
.\.venv\Scripts\python.exe .\generate_qdq_normalization_model.py `
    --pattern add-lpnorm-mul `
    --clip-lpnorm-site image `
    --qdq-profile microsoft
```

## Shapes observed in the original graphs

The following tables were collected from `amd-clip/clip_vit_base_patch16_amd.onnx` and `fp32-gemma4-e2b-it/vision_encoder/model.onnx` after ONNX shape inference. Symbolic dimensions are preserved as exported.

### CLIP normalization shapes

| Operator | Input shape | Parameter shapes | Output shape | Attributes | Count |
|---|---|---|---|---|---:|
| `LayerNormalization` | `[10, 77, 512]` | scale `[512]`, bias `[512]` | `[10, 77, 512]` | axis `-1`, epsilon `1e-5` | 25 |
| `LayerNormalization` | `[1, 197, 768]` | scale `[768]`, bias `[768]` | `[1, 197, 768]` | axis `-1`, epsilon `1e-5` | 25 |
| `LayerNormalization` | `[1, 768]` | scale `[768]`, bias `[768]` | `[1, 768]` | axis `-1`, epsilon `1e-5` | 1 |
| `LpNormalization` | `[1, 512]` | - | `[1, 512]` | axis `-1`, p `2` | 1 |
| `LpNormalization` | `[10, 512]` | - | `[10, 512]` | axis `-1`, p `2` | 1 |

Every listed CLIP normalization input is produced by DQ and every output feeds Q in the source graph.

### Gemma-4-E2B-IT vision normalization shapes

| Operator | Input shape | Parameter/input shapes | Output shape | Attributes | Count |
|---|---|---|---|---|---:|
| `RMSNormalization` | `[batch, num_patches, 12, 64]` | scale `[64]` | `[batch, num_patches, 12, 64]` | axis `-1`, epsilon `1e-6` | 48 |
| `RMSNormalization` | `[batch, num_patches, 768]` | scale `[768]` | `[batch, num_patches, 768]` | axis `-1`, epsilon `1e-6` | 32 |
| `RMSNormalization` | `[batch, _d0, 768]` | scale `[768]` | `[batch, _d0, 768]` | axis `-1`, epsilon `1e-6` | 1 |
| `SkipSimplifiedLayerNormalization` | `[batch, num_patches, 768]` | skip `[batch, num_patches, 768]`, gamma `[768]` | `[batch, num_patches, 768]` | epsilon `1e-6` | 32 |


### Gemma-4-E2B-IT vision model's MatMul shapes for reference
| Gemma 4 E2B-IT vision MatMul use | Left input shape | Right input shape | Output shape | Count |
|---|---|---|---|---:|
| 768-wide projection | `[batch, num_patches, 768]` | `[768, 768]` | `[batch, num_patches, 768]` | 65 |
| Attention QK transpose | `[batch, 12, num_patches, 64]` | `[batch, 12, 64, num_patches]` | `[batch, 12, num_patches, num_patches]` | 16 |
| Attention probabilities by V | `[batch, 12, num_patches, num_patches]` | `[batch, 12, num_patches, 64]` | `[batch, 12, num_patches, 64]` | 16 |
| MLP gate/up projection | `[batch, num_patches, 768]` | `[768, 3072]` | `[batch, num_patches, 3072]` | 32 |
| MLP down projection | `[batch, num_patches, 3072]` | `[3072, 768]` | `[batch, num_patches, 768]` | 16 |
| Pooler | `[batch, _d0, num_patches]` | `[batch, num_patches, 768]` | `[batch, _d0, 768]` | 1 |
| Projector | `[batch, _d0, 768]` | `[768, 1536]` | `[batch, _d0, 1536]` | 1 |

## Run a model

`run_winml_ep.py` measures one provider:

```powershell
.\.venv-winml\Scripts\python.exe .\run_winml_ep.py `
    .\unit-models\clip_visual_dq_matmul_q.onnx `
    --provider cpu `
    --iterations 100
```

Use `--provider vitisai`, `qnn`, or `openvino` for an NPU.

`run_acc.py` compares CPU and NPU outputs:

```powershell
.\.venv-winml\Scripts\python.exe .\run_acc.py `
    .\unit-models\clip_visual_dq_matmul_q.onnx `
    --provider vitisai `
    --seed 1009
```

Add repeatable `--provider-option KEY=VALUE` arguments for provider-specific settings.
Both runners allow CPU fallback by default. Add `--no-cpu-fallback` to require full NPU execution.

## Compile and run a precompiled model

Compile a model for QNN:

```powershell
.\.venv-winml\Scripts\python.exe .\compile_winml_ep_model.py `
    .\unit-models\clip_visual_dq_matmul_q.onnx `
    --provider qnn `
    --output .\unit-models\clip_visual_qnn_ctx.onnx
```

Run the precompiled model in a separate process:

```powershell
.\.venv-winml\Scripts\python.exe .\run_winml_ep.py `
    .\unit-models\clip_visual_qnn_ctx.onnx `
    --provider qnn `
    --iterations 100
```
