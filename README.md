# CCR-AVSS

Implementation of **Controllable Cross-Modal Consistency Repair for Audio-Visual Speech Separation**.

The repository provides a complete waveform-to-waveform system with:

- compressed complex spectral analysis and inverse synthesis;
- an integrated mouth-motion encoder or pre-extracted visual embeddings;
- stage-wise audio-visual representation learning;
- shift-tolerant top-r consistency measurement;
- forward-local reference tracking and selective repair;
- cross-modal gating, bounded residual proposals, and verification;
- alternating time/frequency sequence modeling;
- magnitude-mask and phase reconstruction heads;
- training, evaluation, inference, and JSONL manifest utilities.

## Layout

```text
CCRNet_clean_release/
├── ccr_avss/
│   ├── audio/          # spectral analysis and reconstruction
│   ├── vision/         # mouth-motion encoding and alignment
│   ├── modules/        # consistency repair and T-F modeling
│   ├── models/         # end-to-end separator
│   ├── training/       # objective, metrics, checkpoint helpers
│   ├── data/           # JSONL dataset and batch collation
│   └── utilities/      # configuration and reproducibility helpers
├── commands/           # train, evaluate, and separate entry points
├── tools/              # manifest preparation
├── settings/           # dataset-specific configurations
└── tests/
```

## Installation

```bash
python -m pip install -e .
```

## Data manifest

Each split uses one JSON object per line:

```json
{"id":"sample_0001","mixture":"/path/mix.wav","target":"/path/target.wav","visual":"/path/mouth.npy"}
```

`visual` may contain mouth frames with shape `[T, H, W]` or visual embeddings with
shape `[C, T]`. Set `data.visual_kind` in the YAML configuration accordingly.

## Commands

```bash
python -m commands.train --config settings/lrs2.yaml
python -m commands.evaluate --config settings/lrs2.yaml --checkpoint runs/lrs2/best.pt
python -m commands.separate --config settings/lrs2.yaml --checkpoint runs/lrs2/best.pt \
  --mixture example.wav --visual mouth.npy --output separated.wav
```

## Python interface

```python
import torch
from ccr_avss.models import ConsistencyRepairSeparator

model = ConsistencyRepairSeparator()
mixture = torch.randn(2, 32000)
visual_embedding = torch.randn(2, 512, 50)
estimate = model(mixture, visual_embedding, visual_kind="embedding")
print(estimate.shape)  # [2, 1, 32000]
```

## Citation

```bibtex
@inproceedings{xu2026controllable,
  title     = {Controllable Cross-Modal Consistency Repair for Audio-Visual Speech Separation},
  author    = {Xu, Xinmeng and Xie, Haoran and Qin, S. Joe and Li, Lin and Tao, Xiaohui},
  booktitle = {Proceedings of the ACM International Conference on Multimedia},
  year      = {2026}
}
```
