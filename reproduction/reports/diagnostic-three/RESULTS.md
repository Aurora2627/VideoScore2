# Three-video diagnostic results

Official weights; Windows RTX 4060 Laptop 8 GB; dedicated PyTorch CUDA environment. Valid output means EOS plus three parsed integer scores, not a correct prediction.

Scores are visual quality / text alignment / physical consistency.

| Video | Prediction | Dataset label | Generation (s) | Peak allocated (GiB) |
|---|---|---|---:|---:|
| 004882_h | 3 / 2 / 3 | 4 / 2 / 1 | 140.8 | 5.72 |
| 001033_p | 3 / 3 / 3 | 3 / 2 / 3 | 104.7 | 5.71 |
| 002314_j | 1 / 3 / 2 | 1 / 1 / 1 | 99.3 | 5.72 |

The first video reached EOS but its expanded score labels were rejected by the old parser. Its original result remains unchanged; reparsed-v013.json records the separate parser-only repair with hashes and commit. The other two videos ran using the repaired parser.

This is a reduced-frame, reduced-resolution, quantized inference diagnostic. It does not reproduce paper benchmark metrics, floating-score evaluation, SFT or GRPO. Three videos cannot establish general scoring quality. See saved raw outputs and source snapshots for evidence.