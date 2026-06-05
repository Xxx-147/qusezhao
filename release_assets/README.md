# qusezhao release assets

This folder contains the packaged dataset, model checkpoint, and latest experiment report for the film negative mask-removal project.

## Contents

- `dataset/manifest.csv`: all paired samples
- `dataset/train_manifest.csv`: training split
- `dataset/test_manifest.csv`: test split
- `dataset/train/*_negative.*`: negative inputs
- `dataset/train/*_target.*`: target positives
- `dataset/test/*_negative.*`: held-out negative inputs
- `dataset/test/*_target.*`: held-out target positives
- `models/film_mask_tiny_mixed_true_negative.pt`: current PyTorch checkpoint
- `experiments/latest/03_model_outputs.jpg`: latest visual comparison
- `experiments/latest/metrics.json`: latest test metrics

## Counts

- Total pairs: 71
- Training pairs: 57
- Test pairs: 14

## Quick Use

Install the project first:

```powershell
python -m pip install -e ".[ml]"
```

Run conversion with the included checkpoint:

```powershell
.\.venv-ml\Scripts\python.exe -m film_mask_automation.cli convert input_negative.jpg output_positive.jpg --ai-model release_assets\models\film_mask_tiny_mixed_true_negative.pt
```

AI conversion now defaults to hybrid inference: checkpoint output plus smart rule anchoring, followed by conservative color and contrast restoration. To inspect raw model behavior, add:

```powershell
--no-ai-enhance --no-ai-hybrid
```

Latest 14-sample test snapshot:

- smart auto average MAE: 44.06
- AI hybrid average MAE: 40.48
- target average saturation: 0.1217
- AI hybrid average saturation: 0.1295

Train again using the packaged training split:

```powershell
.\.venv-ml\Scripts\python.exe -m film_mask_automation.ml.train release_assets\dataset\train_manifest.csv models\custom.pt --epochs 8 --steps-per-epoch 60 --batch-size 2 --crop-size 160 --base-channels 32 --device cpu
```

## Attribution

BlueNeg images require the following credit in publication, reproduction, redistribution, or derivatives:

`Copyrighted by Tien-Tsin Wong`

BlueNeg dataset: https://huggingface.co/datasets/ttgroup/blueneg-release
