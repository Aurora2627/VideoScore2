# Reproduction protocol

The model evaluates visual quality, text-to-video alignment, and physical/common-sense consistency. The official model checkpoint and upstream implementation are the baseline; the added work here is a constrained CUDA inference and evidence-saving pipeline.

Start with one of three fixed real benchmark videos. The original generation prompt is the only text input. Ground-truth scores and the metadata thinking field are excluded from model inputs. The diagnostic report compares predictions with labels only after generation. Three samples cannot establish benchmark accuracy or ranking correlation.

The current adapter changes numerical precision (NF4), frame sampling, resolution and decoding relative to the official demo; its 1024-token generation cap matches the demo limit. It enables generation KV caching and does not retain full-vocabulary per-token logits. It reports validated hard scores. Treat these as implementation differences, not as a reproduction of the paper metrics.

The official demo floating-score expression is not a calibrated confidence score. This adapter does not claim floating-score equivalence. An audit of the benchmark and floating-score dispatch is required before comparing floating-score metrics.

Read run result.json, raw_output.txt, source snapshots and environment information together. A failed parse, truncated generation, exception or out-of-memory run must remain recorded. Do not replace an invalid score with a plausible number. Any later improvement should be compared on a separate validation split, with the final test split reserved for final evaluation.

No SFT or GRPO training has been completed in this reproduction. All claims must distinguish environment checks, quantized inference, diagnostic comparisons, full benchmark evaluation and training.
