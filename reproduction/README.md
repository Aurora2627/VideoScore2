# VideoScore2 Windows CUDA reproduction

Upstream: TIGER-AI-Lab/VideoScore2, commit 9e0fcbaeee0283b03efaaf8cd906f6381a808c75. Original authors retain credit. Upstream scripts remain unchanged.

Status: dedicated Windows virtual environment, CUDA/NF4 kernel, and three real input videos validated. Official model weights are downloaded and SHA256 validated. The initial 256-token attempt was truncated before final scores; no successful complete score or benchmark result yet.

The additional infer_cuda.py is an experimental 8 GB GPU adaptation: 4-bit NF4, double quantization, SDPA, uniform capped frames, reduced resolution, greedy generation. It extracts the original demo prompt without importing the upstream script. It reports hard 1-5 scores only, with strict validation. These changes do not reproduce the paper's full evaluation or floating-score protocol. No claim of numerical equivalence is made.

Use Python 3.11 in a dedicated environment. Install torch and torchvision from the official cu124 index, then the remaining pinned dependencies. Run infer_cuda.py --help for parameters. A fixed model commit, real input video, its generation prompt, and a new output directory are mandatory. The output contains source snapshots, input hash, sampling indices, environment, raw generation, token IDs, timing and GPU memory. Weights and private videos must remain outside Git.

Training and full benchmark reproduction are out of scope for this initial inference validation.

## Installation and verification

Create a dedicated Python 3.11 virtual environment. Always invoke its Python executable, including for pip. Install requirements-torch-cu124.txt first, then requirements-windows.txt. Run check_cuda.py with an output JSON path, followed by pip check. The CUDA check executes a real NF4 kernel; it does not load the VideoScore2 model.

The local VS Code workspace outside this repository points to the dedicated interpreter. The integrated shell may still display an inherited Conda base prompt; sys.executable and sys.prefix are the evidence for which interpreter runs a command.

Run test_protocol.py for score parsing and prompt extraction tests. Use prepare_subset.py only with pinned official metadata and metadata-provenance.json. Use run_diagnostic.py with subset.json, the local model snapshot, immutable model revision, and a new output directory. Weight download and GPU validation must finish first. All model inputs exclude labels and teacher explanations.

See provenance.json for pinned sources and PROTOCOL.md for evaluation limits. No successful model inference is claimed until saved outputs have been inspected.

The generation budget now defaults to 1024 tokens, matching the official demo limit. Score parsing uses the final answer after a closed thinking block. Truncated generations are always rejected even if some scores appear. The first truncated attempt remains in reports/initial-truncated-attempt.json; full local run artifacts are preserved separately.
