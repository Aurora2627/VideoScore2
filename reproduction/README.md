# VideoScore2: Windows CUDA inference reproduction

This fork preserves TIGER-AI-Lab's VideoScore2 implementation and attribution. Upstream commit: 9e0fcbaeee0283b03efaaf8cd906f6381a808c75. The model, training method, dataset and paper results belong to their original authors.

## What this reproduction adds

- Dedicated Python 3.11 / PyTorch 2.6.0+cu124 environment with pinned dependencies and a real NF4 CUDA kernel check.
- An 8 GB GPU inference path using 4-bit NF4 double quantization, BF16 computation, SDPA, four uniformly sampled frames and a 224-pixel longest side.
- Original demo prompt extraction, greedy generation with a 1024-token cap, EOS validation, and parsing of three integer scores after the thinking block.
- Fixed official inputs with model/video hashes, source snapshots, environment, raw outputs, token IDs, latency and allocated/reserved GPU memory saved per run.
- Three real diagnostic videos. See [results](reports/diagnostic-three/RESULTS.md) and [machine-readable evidence](reports/diagnostic-three/summary.json).

These are inference and reproducibility contributions. No SFT, GRPO training or full paper benchmark reproduction was performed. Reduced frames, resolution, quantization and hard-score parsing differ from the paper's evaluation; no numerical equivalence is claimed. Valid output means complete, parseable output, not a correct prediction.

## Environment

Create a dedicated Python 3.11 virtual environment. Use its Python executable for every command, including package installation. Install requirements-torch-cu124.txt first, then requirements-windows.txt. requirements-lock-windows.txt records the environment actually tested. Do not install these packages into a shared Conda base environment.

Run check_cuda.py with an output JSON path and run pip check. CUDA validation executes a real bitsandbytes NF4 kernel. Run test_protocol.py for the twelve parsing/prompt checks. check_inputs.py and check_processor.py verify the actual videos and Transformers processor. Consult each script's --help for its arguments.

The local VideoScore2.code-workspace selects the dedicated virtual environment and provides a CUDA debug configuration. In an integrated shell an inherited base prompt can remain visible; sys.executable and sys.prefix identify the actual interpreter. Always use a new experiment output directory when debugging or rerunning.

## Inputs and inference

Use prepare_subset.py with pinned official metadata and metadata-provenance.json. The fixed manifest includes the first three samples; there is no quality-based selection. Labels and teacher explanations are never passed to the model. Weights and video files remain outside Git.

Use run_diagnostic.py with --manifest, --model (local snapshot), --revision (immutable model commit), --output-dir (new directory), --count 3, --frames 4 and --max-new-tokens 1024. It invokes infer_cuda.py using the same interpreter. Run infer_cuda.py directly for a single video, supplying its original generation prompt. Both scripts refuse to overwrite experiment directories. The diagnostic uses local/offline weights.

Model revision: 09a2732cb64fa566a1f332f978368292ce5c295c. All four official weight shards were checked against published SHA256 hashes; see reports/model-snapshot.json. See provenance.json and PROTOCOL.md for source and protocol details.

## Observed issues and audit trail

The initial 256-token attempt ended inside the thinking block and was rejected. Its record is preserved in reports/initial-truncated-attempt.json. At 1024 tokens, the first video reached EOS but added descriptors to its score labels. A regression fixture and parser repair address that format. reparse_saved.py wrote a separate, hashed parser-only result without rerunning the model or changing the original unsuccessful result. The other two videos used the repaired parser.

The saved result for the first video therefore retains its original invalid_output status; reparsed-v013.json and the combined summary explicitly record the repair. No failed artifact is overwritten and no model result is invented.
