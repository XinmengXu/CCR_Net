# CCRNet

Official code for **Controllable Cross-Modal Consistency Repair for Audio-Visual Speech Separation**, accepted by ACM Multimedia 2026.

CCRNet is a complete waveform-to-waveform audio-visual speech separation model. The implementation contains:

- STFT analysis and inverse-STFT reconstruction;
- magnitude, phase, and visual feature encoders;
- stage-wise audio-visual representation learning;
- shift-tolerant cross-modal consistency scoring;
- reference-guided CCR triggering;
- cross-modal gating, bounded residual correction, and accept-if-improved verification;
- time-frequency Transformer modeling;
- magnitude and phase decoders;
- training, evaluation, preprocessing, and inference entry points.

## Project structure

```text
CCRNet/
├── configs/                    # LRS2, LRS3, and VoxCeleb2 configurations
├── look2hear/
│   ├── datas/                  # AVSS datasets and data modules
│   ├── losses/                 # CCRNet multi-level objective
│   ├── models/                 # Complete CCRNet AVSS backbone
│   ├── system/                 # PyTorch Lightning training system
│   └── videomodels/            # Frozen visual frontend
├── preprocess/                 # Dataset metadata preparation
├── train.py
├── evaluate.py
├── inference.py
└── check_model.py
```

## Basic commands

```bash
pip install -r requirements.txt
python check_model.py
python train.py --config configs/LRS2-CCRNet.yml
python evaluate.py --config configs/LRS2-CCRNet.yml \
  --checkpoint experiments/LRS2-CCRNet/CCRNet.pth
```

The visual frontend checkpoint is configured through `video_frontend.pretrain`. Dataset metadata directories contain `mix.json`, `s1.json`, and `s2.json`, following the preprocessing scripts in `preprocess/`.

## Model interface

```python
import torch
from look2hear.models import CCRNet

model = CCRNet()
mixture = torch.randn(2, 32000)          # [B, T]
visual = torch.randn(2, 512, 50)         # visual frontend embeddings
estimate = model(mixture, visual)         # [B, 1, T]
```

## Citation

```bibtex
@inproceedings{xu2026ccrnet,
  title     = {Controllable Cross-Modal Consistency Repair for Audio-Visual Speech Separation},
  author    = {Xu, Xinmeng and Xie, Haoran and Qin, S. Joe and Li, Lin and Tao, Xiaohui},
  booktitle = {Proceedings of the ACM International Conference on Multimedia},
  year      = {2026}
}
```
