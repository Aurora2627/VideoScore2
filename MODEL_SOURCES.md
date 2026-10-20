# Model and dependency sources

ClipRank uses the VideoScore2 checkpoint as its scoring backend. The checkpoint, datasets and original model training are credited to their respective authors.

- Model implementation: https://github.com/TIGER-AI-Lab/VideoScore2
- Model weights: https://huggingface.co/TIGER-Lab/VideoScore2
- Dataset: https://huggingface.co/datasets/TIGER-Lab/VideoFeedback2
- Model checkpoint commit: 09a2732cb64fa566a1f332f978368292ce5c295c
- Pinned model implementation commit: 9e0fcbaeee0283b03efaaf8cd906f6381a808c75

The repository LICENSE and CITATION.cff are retained. Refer to the original model and dataset pages for their own terms; the source-code license alone does not determine model or dataset terms. The original model documentation is retained in docs/model-reference.md. Additional dependency versions are recorded in the environment lock file.

Application contributions are resident CUDA batch inference, input integrity checks, prompt-grouped candidate screening, configurable review rules and auditable HTML/JSON reports. Training the scoring model and the original paper's results are not claimed as application contributions.
