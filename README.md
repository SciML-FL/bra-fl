# Bayesian Robust Aggregation for Federated Learning (BRA-FL)

This repository contains the official implementation of the paper:

<div align="center">
<h2 style="border-bottom:0px none #000; margin-bottom: 0;"><a href="https://www.arxiv.org/abs/2505.02490">Bayesian Robust Aggregation for Federated Learning</a></h2>

  <p style="border-bottom:0px none #000; margin-bottom: 0;">
    <a href="https://scholar.google.com/citations?user=XNBDwTkAAAAJ&hl=en">Aleksandr Karakulev</a> | 
    <a href="https://usamazf.github.io/">Usama Zafar</a> | 
    <a href="https://scholar.google.se/citations?user=PIGlWyYAAAAJ&hl=en">Salman Toor</a> | 
    <a href="https://www.prashantsingh.se/">Prashant Singh</a>
  </p>
  <p><a href="https://www.uu.se/en">Uppsala University</a></p>
</div>

The current implementation contains the federated-learning runtime and compact experiment definitions for Bayesian robust aggregation. It runs on an ordinary workstation with Python; Slurm, Apptainer, and cluster accounts are not required. CPU execution is supported, while the complete image experiments require substantial time and memory and normally benefit from a GPU.

## Get the code

```sh
git clone https://github.com/SciML-FL/bra-fl.git
cd bra-fl
```

## Install

Use Python 3.11 and run commands from the repository root. Create an environment:

```sh
python -m venv .venv
```

Activate it with `source .venv/bin/activate` on Linux/macOS, or `.venv\Scripts\Activate.ps1` in Windows PowerShell. For a CPU installation:

```sh
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
```

For CUDA, install the matching PyTorch 2.5.1 / torchvision 0.20.1 build for your CUDA platform using the [official PyTorch instructions](https://pytorch.org/get-started/previous-versions/), then install `requirements.txt`. The requirements describe the validated CPU environment for this implementation. The exact original cluster environment has not been recovered, so this is not a claim of bitwise reproduction of published results across hardware.

`requirements-tested-cpu.txt` records all installed packages in the Windows CPU validation environment. After installing the CPU PyTorch wheels, it can be installed instead of `requirements.txt` to pin the transitive dependencies as well.

## Quick checks

```sh
python smoke_test.py
python run_sweep.py --dry-run
```

The smoke check uses synthetic data for one round of Bayesian aggregation on CPU, with no dataset download. It checks execution, result generation, and handling of a deliberately failing client; its output is not an experimental result for the paper. It retains tiny artifacts in `work/smoke-test`; use `--output-root work/smoke-test-2` for a second run. The dry run validates the complete configuration matrix without training.

## Run experiments locally

Start with one configuration from the smaller regression experiment:

```sh
python run_sweep.py --only c1-parkinson --limit 1 --device cpu --workers 1 --download --output-root work/parkinson-example
```

This still uses the experiment's full training settings. `--limit` limits the number of configurations, not training rounds. Each invocation requires an absent or empty output directory; completed results are never silently overwritten or treated as a resumable partial run.

For a CIFAR-10 subset on a CUDA device:

```sh
python run_sweep.py --only c1-cifar10 --limit 1 --device cuda:0 --workers 1 --download --output-root work/cifar10-example
```

Run the complete matrix after preparing all datasets:

```sh
python run_sweep.py --device cuda:0 --workers 1 --data-root work/data --output-root work/full-suite
```

`--only` may be repeated to select multiple blocks. Configurations run sequentially; `--workers` controls concurrent clients within one configuration. Start with one worker. Models are kept in memory for multiple clients, so limiting workers alone does not guarantee a small memory footprint. CUDA availability and device indices are checked before training.

The runner writes effective configurations, process logs, a run status file, model weights, and experiment history below the output root. A completion marker is written only after the subprocess and required result checks pass. An execution or missing-result failure stops the sweep with exit code 1. Numerical divergence is retained separately, receives no completion marker, and makes the batch return exit code 2. Client processes and seed handling retain the supplied implementation's ProcessPool execution path.

## Experiment matrix

| Block | Selector | Configurations | Experiment IDs |
|---|---|---:|---|
| C1, CIFAR-10 | `c1-cifar10` | 378 | 1–378 |
| C1, CIFAR-100 | `c1-cifar100` | 378 | 379–756 |
| C1, Tiny ImageNet | `c1-tiny-imagenet` | 378 | 757–1134 |
| C1, Parkinsons | `c1-parkinson` | 294 | 1135–1428 |
| C2, misspecification | `c2-misspecification` | 234 | 1429–1662 |
| C3, client subsampling | `c3-subsampling` | 54 | 1663–1716 |
| C4, intermittent attacks | `c4-intermittent` | 66 | 1717–1782 |
| C5, main-page plot | `c5-main-page-plot` | 2 | 1783–1784 |

The full matrix contains **1,784 configurations**. Definitions are in `papers/p02_bayesian_aggregation/experiments/2026_bayesian/`. The manifest records contiguous IDs and expected counts; templates supply complete defaults and sweeps override them. Coupled YAML axes use the custom safe loader in `fedml/configs/parser.py`. In particular, an explicitly assumed malicious-client count of zero in C2 is retained as zero.

You can also generate configurations without launching training:

```sh
python -m tools.experiments.build_suite papers/p02_bayesian_aggregation/experiments/2026_bayesian/suite.yaml --dry-run --quiet
```

## Datasets and weights

Datasets and pretrained weights are not included in this repository.

- **CIFAR-10 and CIFAR-100:** `--download` enables torchvision's dataset download. Both use the `cifar` subdirectory of `--data-root`.
- **Parkinsons Telemonitoring:** use `--download` for the initial fetch of UCI dataset 189 through `ucimlrepo`. It is cached as `parkinsons/uciml_parkinsons.joblib`. The target is `motor_UPDRS`. The local runner requires this cache when `--download` is omitted.
- **Tiny ImageNet:** obtain `tiny-imagenet-200` from the dataset provider. Put its `train` directory under `work/data/tiny-imagenet-200/train`. The loader expects validation images arranged by class. Preserve the supplied raw validation directory outside that destination and convert it with:

```sh
python -m tools.data.setup_tiny_imagenet path/to/raw/tiny-imagenet-200/val work/data/tiny-imagenet-200/val
```

The Tiny ImageNet loader does not download data. Its ConvNeXt model loads torchvision's pretrained weights on first use, even though the separate `WEIGHT_PATH` configuration is null. Internet access or a populated PyTorch weight cache is therefore required. This behavior is retained from the experiment implementation.

## Acknowledgements

This work was supported by Uppsala University. The computations were enabled by resources provided by the National Academic Infrastructure for Supercomputing in Sweden (NAISS), partially funded by the Swedish Research Council through grant agreement no. 2022-06725. We thank the open-source community for tools and baselines used in this project.

## Contact

For questions or collaborations:

- Aleksandr Karakulev — [aleksandr.karakulev@it.uu.se](mailto:aleksandr.karakulev@it.uu.se)
- Usama Zafar — [usama.zafar@it.uu.se](mailto:usama.zafar@it.uu.se)

## Citation

If you use this work, please cite:

```bibtex
@article{karakulev2025bayesianrobustaggregationfederated,
      title={Bayesian Robust Aggregation for Federated Learning}, 
      author={Aleksandr Karakulev and Usama Zafar and Salman Toor and Prashant Singh},
      year={2025},
      eprint={2505.02490},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.02490}, 
}
```

## License notices

The existing public repository license is retained unchanged in [LICENSE](LICENSE). The imported runtime and experiment package also carry the following MIT notice, preserved from their source. This notice does not replace the license of the earlier public release.

```text
MIT License

Copyright (c) 2024 Usama Zafar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
