# v4 static and ClusterChain results: MANIFEST

**Status:** both suites are complete and pass the audit, with every protected check passing. `report.py` wrote all figures and tables.

## Code
- **Experiments:** run at commit `511deab9255aa6e127cafa0b1f8d64dc29747d33` (branch `exp/review-null-refine-clusterchain`).
- **Report:** generated at commit `198022d0e41c9f2f918687820a18369f3cc82201`.
  - The only change from `511deab` is in `experiments/report.py`: a refinement with no split design now completes at its trigger event.
  - `report.py` is not covered by `code_identity`.
  - `run.code_identity()` at `198022d` equals the stored value.
- **`code_identity`:** all 1,470 `unit.json` files carry `7de896507ae118b7d1384450b0b083dfe4394f072662ac60d2a136e4cd1e4137`.
- **`analysis_source_sha256`:** `d8f24e19210b3929b3c871bdba8d6b5f33bb5bd2f292f2e45bdc97ec5a23b835`.
- **Protocol:** `matched-controlled-cbo-v4`, seed namespace `matched-controlled-noisy-v2`.

## Machine
- **Date:** 2026-10-03. Static ran 16:56–17:09 CEST and ClusterChain 17:09–18:23 CEST. The report ran at 18:44 CEST.
- **Hardware:** AMD Ryzen 9 9950X 16-Core Processor (16 physical cores, 32 threads), 123 GB RAM.
- **OS:** Ubuntu 24.04.2 LTS, kernel 6.17.0-35-generic.
- **Python:** 3.10.21 (conda-provided interpreter) in venv `.venv-static`, reused from the v3 run. `requirements-static.txt` is unchanged since v3.

## Tests (at `511deab`)
- `.venv-static/bin/python -m pytest -q`: 228 passed, 37 skipped, 9 deselected.
- `MATCHED_INTEGRATION=1 ... test_matched_protocol.py test_matched_refinement.py test_matched_fallback.py`: 20 passed.
- Smoke check (`results/smoke_v4`, both suites): `"failures": []`. Protected checks matched 2/2 (static) and 7/7 (ClusterChain).

## Commands
The suites ran sequentially in tmux with `--jobs 14` (16 physical cores − 2), each wrapped in `/usr/bin/time -v`.
```
.venv-static/bin/python experiments/run.py --suite static --jobs 14 --outdir results/v4
.venv-static/bin/python experiments/run.py --suite clusterchain --jobs 14 --outdir results/v4
.venv-static/bin/python experiments/analyze.py --suite static --results results/v4 --out results/v4/static.json
.venv-static/bin/python experiments/analyze.py --suite clusterchain --results results/v4 --out results/v4/clusterchain.json
# At 511deab, report.py failed with IndexError (report.py:118, empty split_event_ids). It was fixed in 198022d; nothing was rerun.
.venv-static/bin/python experiments/report.py --results results/v4 --out results/v4/paper_outputs --suites static clusterchain
cd results/v4/paper_outputs/figures && for f in *.pdf; do pdftoppm -png -r 160 -singlefile $f ${f%.pdf}; done
tar -czf results_v4_raw.tar.gz -C results/v4 paper
```

The only line `report.py` printed apart from the file list:
```
HQCBO refinement completes at cost [12] (Figure 3 marks the median)
```

| Suite | Units | Wall clock | CPU (`/usr/bin/time` user + sys) | Sum of unit `wall_seconds` | Peak RSS (one process) |
|---|---|---|---|---|---|
| static | 630/630 | 771 s (12 min 51 s) | 10,771 s (9,496 + 1,275) | 9,992 s | 0.95 GB |
| clusterchain | 840/840 | 4,452 s (1 h 14 min 12 s) | 62,194 s (61,918 + 276) | 61,352 s | 1.50 GB |

## Audit
| Suite | Status | Valid units | Failures | Protected checks |
|---|---|---|---|---|
| static | `complete` | 630/630 | `[]` | 60/60 `exact_equal: true` |
| clusterchain | `complete` | 840/840 | `[]` | 210/210 `exact_equal: true` |

## Replaced units
None. No protected check failed, so no unit was rerun or replaced.

## Raw results
`results_v4_raw.tar.gz` (in this folder) archives `results/v4/paper/`: 1,470 unit folders, 52,489,474 bytes.

## pip freeze (`.venv-static`)
The freeze was taken after pulling `198022d`. The editable `ccbo` line therefore shows that commit, although the experiments ran at `511deab`. `ccbo/` is identical in both commits.
```
ananke-causal==0.5.0
annotated-types==0.8.0
-e git+ssh://git@github.com/fmfsa/ccbo.git@198022d0e41c9f2f918687820a18369f3cc82201#egg=ccbo&subdirectory=../../.claude/worktrees/pr9-experiment-runs-c22831
certifi==2026.7.22
cffi==2.1.1
charset-normalizer==3.5.2
cloudpickle==3.1.2
contourpy==1.3.2
cryptography==50.0.2
cuda-bindings==13.4.3
cuda-pathfinder==1.8.3
cuda-toolkit==13.0.3.0
cycler==0.12.1
Cython==3.3.0
decorator==5.3.1
dill==0.4.1
emcee==3.1.6
emukit==0.5.1
exceptiongroup==1.3.1
filelock==4.0.9
fonttools==4.65.0
fsspec==2026.9.0
google-ai-generativelanguage==0.6.15
google-api-core==2.33.0
google-api-python-client==2.201.0
google-auth==2.59.1
google-auth-httplib2==0.4.4
google-generativeai==0.8.6
googleapis-common-protos==1.75.0
GPy==1.13.2
graphviz==0.21
grpcio==1.84.0
grpcio-status==1.71.2
httplib2==0.32.0
idna==3.20
iniconfig==2.3.0
jax==0.4.38
jaxlib==0.4.38
Jinja2==3.1.6
joblib==1.6.0
kiwisolver==1.5.1
klepto==0.2.8
MarkupSafe==3.0.3
matplotlib==3.10.9
ml_dtypes==0.5.4
mpmath==1.3.0
mystic==0.4.5
networkx==3.4.2
numpy==1.26.4
nvidia-cublas==13.1.1.3
nvidia-cuda-cupti==13.0.85
nvidia-cuda-nvrtc==13.0.88
nvidia-cuda-runtime==13.0.96
nvidia-cudnn-cu13==9.24.0.43
nvidia-cufft==12.0.0.61
nvidia-cufile==1.15.1.6
nvidia-curand==10.4.0.35
nvidia-cusolver==12.0.4.66
nvidia-cusparse==12.6.3.3
nvidia-cusparselt-cu13==0.8.1
nvidia-nccl-cu12==2.32.3
nvidia-nccl-cu13==2.30.7
nvidia-nvjitlink==13.4.92
nvidia-nvshmem-cu13==3.4.5
nvidia-nvtx==13.0.85
opt_einsum==3.4.0
packaging==26.3
pandas==1.5.3
paramz==0.9.6
patsy==1.0.3
pgmpy==0.1.26
pillow==12.3.0
pluggy==1.6.0
pox==0.3.7
proto-plus==1.28.2
protobuf==5.29.6
pyasn1==0.6.4
pyasn1_modules==0.4.2
pycparser==3.0
pydantic==2.13.5
pydantic_core==2.46.5
Pygments==2.21.0
pygraphviz==2.0.3
pyparsing==3.3.3
pytest==9.1.1
python-dateutil==2.9.0.post0
pytz==2026.4
PyYAML==6.0.3
requests==2.34.2
scikit-learn==1.7.2
scipy==1.12.0
seaborn==0.13.2
six==1.17.0
statsmodels==0.13.5
sympy==1.14.0
threadpoolctl==3.7.0
tomli==2.4.1
torch==2.14.1
tqdm==4.70.1
triton==3.8.0
typing-inspection==0.4.4
typing_extensions==4.16.0
uritemplate==4.2.0
urllib3==2.8.0
xgboost==3.2.0
```

## SHA-256 of committed files
Paths are relative to `experiment_outputs/v4/`. The list excludes this file.
```
039a4095434a2325bd8768400eec76215cc73e3ad0de5c1a7ca323ba822f8a40  clusterchain.csv
4766dfab981873de5942528ef9df6ab13cc44d8a5dcfbe3302517cbb8e6b790c  clusterchain.json
07d718284ce9d88ce1d05977265fc1f925cd03df48c93d6cfb6dc2c8821eb634  paper_outputs/figures/clusterchain.pdf
ca3f120782058643d186188b6210538267a035aef975c6249ec5906ec7dfd3fd  paper_outputs/figures/clusterchain.png
87187291fa7a6e4a518b11eaeb1f93c80a3dff148c3ed83cc9a0545a5b3c20f7  paper_outputs/figures/cost_indexed.pdf
1901ea08c7d10fe6016c9a9d86f84c8933b3acecb99ef29991bcb2e9dce38f10  paper_outputs/figures/cost_indexed.png
f8c727f3575367640faf9b6f97970d12590940b04af003281de4d6b9fbb4fcb7  paper_outputs/figures/minimal_refine.pdf
bca139862e88f7ce5501750b058b4ded77603e4a635b4ff25b1c56272cb81fa8  paper_outputs/figures/minimal_refine.png
9fd440251e1a9ab65f5a5a1b013542032d7946546b270cc80447380c5e0b446e  paper_outputs/figures/trial_indexed.pdf
50a1606434a8e656724786d9544b6c269e4ca65c13700a43a92f14ad17dc2f2c  paper_outputs/figures/trial_indexed.png
e0871ba1612a50e243e08253ddb7cbc63b74e44c0a050c3fa9734904bc8218df  paper_outputs/tables/clusterchain_partitions.tex
38b64d064e1b3a70be24b75bced74acbbde398b5cf0d15ab7c01efeed5aee82e  paper_outputs/tables/clusterchain_results.tex
3eebac5ab50106df1131e6ca5b645b93b2de1309b978faae2985629ee265f937  paper_outputs/tables/cost_accounting.tex
bb441acb7e0937bbdd41664151048cb185515bc2b4ed3474304aed8f5ced88bb  paper_outputs/tables/minimal_ablations.tex
44af193e5ec624b04a1498666f89bb4b2b0ff8f70ceea1e3f4ee3b5aa0a60d4a  paper_outputs/tables/minimal_taxonomy.tex
d6b527ce4cf270fc83e8614597d0a7508172f1cce72311a086cc968d214b9d07  results_v4_raw.tar.gz
539a7357f7a9142b11543f7af4e1a2cc70905be4f8ea9540b1fb86a5715fd689  static.csv
6e0f573669f1debeebcae7f656b2e78d6cd2dc1ce926e078b2eedb7d6daac263  static.json
```
