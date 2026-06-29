# ML Fault Attacks on TinyML

Code and evaluation artifacts for the paper [**"Fault Injection Attacks and Countermeasures on TinyML Algorithms"**](https://ieeexplore.ieee.org/document/11604834) **(HOST 2026 Best Student Paper).**

This repository contains the experimental setup, attack scripts, target firmware, and protection implementations used to evaluate voltage and electromagnetic (EM) fault injection attacks against four representative TinyML workloads, along with Randomized Self-Reduction (RSR) based countermeasures.

## Overview

TinyML deploys machine-learning inference directly on resource-constrained microcontrollers, but the physical accessibility of these devices makes them attractive targets for fault injection. This work shows that carefully tuned voltage glitches can:

- Reduce inference accuracy across all four models studied,
- In some configurations, steer the model toward a *fixed*, attacker-chosen label rather than producing random errors,
- And do so without triggering a system reset (i.e., remain stealthy).

The repository also implements RSR-based countermeasures that wrap the dominant linear-algebra kernels (sigmoid, vector–matrix multiply, convolution) in randomized, redundantly evaluated, median-aggregated computations.

This repository contains the **ChipWhisperer-Husky voltage-glitching** experiments against the SAM4S target. The Pinata-board voltage glitching and EM fault injection experiments described in the paper used different hardware and tooling and are not included here.

## Repository Contents

```
.
├── firmware/                                     # On-device C firmware for the SAM4S target
│   ├── fastgrnn/                                 # FastGRNN inference (unprotected and RSR-protected)
│   ├── image-cnn/                                # TinyCNN MNIST inference (unprotected and RSR-protected)
│   ├── logistic-regression/                      # Logistic Regression inference (unprotected and RSR-protected)
│   ├── wakeWord-cmsisnn/                         # Wake Word detection inference (unprotected and RSR-protected)
│   ├── cmsis_nn/                                 # CMSIS-NN kernels (used by TinyCNN and Wake Word)
│   ├── hal/                                      # Hardware abstraction layer (ChipWhisperer)
│   ├── simpleserial/                             # SimpleSerial protocol for host ↔ target UART
│   ├── tiny-ml/                                  # Shared TinyML build target / top-level project
│   └── Makefile.inc                              # Shared build configuration
├── notebooks/
│   ├── tiny-ml-fast-grnn.ipynb                   # FastGRNN — ChipWhisperer-Husky voltage-glitch attack
│   ├── tiny-ml-image-cnn.ipynb                   # TinyCNN — ChipWhisperer-Husky voltage-glitch attack
│   ├── tiny-ml-logistic-regression.ipynb         # Logistic Regression — ChipWhisperer-Husky attack
│   ├── tiny-ml-wakeword.ipynb                    # Wake Word — ChipWhisperer-Husky attack
│   ├── Hamming_distance.ipynb                    # Hamming distance analysis
│   ├── Intermediate_analysis.ipynb               # Intermediate-value analysis
│   ├── Confusion_matrices_with_reset_rate.ipynb  # Aggregate confusion matrices and reset-rate plots
│   └── utils.py                                  # Shared helpers used by the notebooks
├── requirements.txt                              # Pinned host-side Python dependencies
└── README.md
```

The four `tiny-ml-*.ipynb` notebooks are the **ChipWhisperer-Husky voltage-glitching attack drivers**, one per model. They run the two-stage parameter search (Optuna sweep → fine-grained timing sweep) and log per-trial outcomes.

`Hamming_distance.ipynb` reproduces the average hamming weight analysis from the paper: it analyses the bit level effects in the layer output.

`Intermediate_analysis.ipynb` reproduces the fault-propagation analysis from the paper: it compares per-layer outputs to the no-fault baseline and computes effective-error rates. 

`Confusion_matrices_with_reset_rate.ipynb` aggregates results across configurations to produce the figures in Section V.

## Target Models

Four TinyML workloads are evaluated, spanning tabular, audio, vision, and recurrent inference:

| Model | Task | Format | Key kernel attacked |
|---|---|---|---|
| **FastGRNN** | Sequence classification (16 × 16) | float, [Microsoft EdgeML](https://github.com/microsoft/EdgeML) | Matrix–vector multiply in the recurrent layer |
| **TinyCNN** | MNIST digit classification | int8 (TFLite and CMSIS-NN) | First convolution layer |
| **Logistic Regression** | Binary classification (11 features) | Custom lightweight C | Weight–feature MAC and sigmoid |
| **Wake Word** | Keyword spotting on MFCC features | int8 (TFLite and CMSIS-NN) | First fully-connected layer |

Each model exposes a firmware-defined trigger pin around its Region of Interest (ROI), which the glitching hardware uses to align fault injection with the target computation.

## Hardware Setup (ChipWhisperer-Husky path)

The notebooks in this repo target the following setup:

- **ChipWhisperer-Husky** — glitch generator and capture
- **CW313 interposer** — baseboard providing power, UART (TIO1/TIO2), and trigger routing (GPIO4 / nRSTOUT)
- **CW312T-SAM4S target** — ATSAM4S2A microcontroller (Cortex-M4)
- Host PC running the notebooks over USB

Wiring follows the standard ChipWhisperer crowbar-glitching configuration: the Husky drives a crowbar onto the target's VCC rail, with the trigger sourced from a firmware-toggled GPIO that brackets the model's ROI.

## Software Requirements

The host-side notebooks were developed and run with:

- **Python 3.11** (tested with 3.11.9)
- **[`chipwhisperer`](https://github.com/newaetech/chipwhisperer) 6.0.0** — Husky capture/glitch API
- `optuna` — Stage 1 glitch-parameter search
- `numpy`, `pandas`, `matplotlib`, `seaborn` — analysis and plotting
- `tqdm` — progress reporting
- `jupyterlab` — to run the notebooks

Exact pinned versions are in [`requirements.txt`](requirements.txt).

Building the target firmware additionally requires an ARM bare-metal toolchain
(`arm-none-eabi-gcc`; tested with 12.2). The int8 models ship with pre-generated
weights, so TensorFlow / TensorFlow Lite Micro are only needed if you want to
retrain and re-quantize the models from scratch.

### Setup (as used for the experiments)

```bash
# 1. Python environment — pyenv with a dedicated virtualenv named "cw"
pyenv install 3.11.9
pyenv virtualenv 3.11.9 cw
pyenv activate cw

# 2. Host-side Python dependencies
pip install -r requirements.txt

# 3. Hardware access (Linux): install NewAE's udev rules so the Husky is
#    reachable over USB without root. The 50-newae.rules file ships with
#    ChipWhisperer (see the newaetech/chipwhisperer repo). Then replug the device.
sudo cp 50-newae.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

## Running an Attack

1. **Build and flash the target firmware** for the model you want to attack. Each model lives in its own subdirectory under `firmware/` (`fastgrnn/`, `image-cnn/`, `logistic-regression/`, `wakeWord-cmsisnn/`) and includes both an unprotected and an RSR-protected variant. Use the ChipWhisperer programmer to flash the resulting binary onto the CW312T-SAM4S target.
2. **Connect the host PC to the Husky** over USB and verify the target responds over UART (the firmware uses ChipWhisperer's SimpleSerial protocol).
3. **Open the corresponding attack notebook** under `notebooks/` — `tiny-ml-fast-grnn.ipynb`, `tiny-ml-image-cnn.ipynb`, `tiny-ml-logistic-regression.ipynb`, or `tiny-ml-wakeword.ipynb`. Each notebook is organized as:
   - Setup: scope/target initialization, glitch-module configuration
   - Stage 1: Optuna-driven sweep over `(width, offset, ext_offset, repeat)`, scoring configurations by misprediction count subject to a reset-rate cap
   - Stage 2: fine-grained sweep of `ext_offset` within the ROI at the best Stage 1 configuration(s)
   - Logging: per-trial classification and persistence to disk
4. **Run cells in order.** Trial outcomes are saved so sweeps can be resumed without losing prior results.
5. **Reproduce the analysis figures** by running `Intermediate_analysis.ipynb`, `Hamming_distance.ipynb` and `Confusion_matrices_with_reset_rate.ipynb` against the captured trial data.

## Countermeasures (RSR)

Each of the four models ships with both an unprotected and an **RSR-protected** firmware variant under `firmware/`. The protected variants wrap the dominant linear-algebra kernels with randomized, redundantly evaluated, median-aggregated computations:

- Logistic regression (linearity and a sigmoid RSR identity)
- Vector–matrix multiplication (bilinear decomposition into four random/perturbed sub-products)
- Convolution (im2col reduction onto the protected vector–matrix primitive)

Each protected kernel evaluates `T` randomized repetitions and aggregates results with element-wise median voting. Across all four models, clean-input accuracy is preserved within ~1% of the unprotected baseline; runtime overhead ranges from ~1.6× (TinyCNN) to ~34× (FastGRNN, which invokes the protected matvec once per recurrence step). See Tables V and VI of the paper.

RSR is designed to tolerate transient computation faults; persistent architectural-state corruption (e.g., corrupted SRAM/register state across inferences) requires complementary defenses such as state checksums.

## Acknowledgements

This work was supported in part by the U.S. National Science Foundation under grants [2245344](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2245344), [CCF-2153748](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2153748), and [CNS-2442993](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2442993); by the [Commonwealth Cyber Initiative](https://cyberinitiative.org/); and by the [Air Force Office of Scientific Research](https://www.afrl.af.mil/AFOSR/) under award FA9550-22-1-0548. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the sponsors.

## License

This project's own code is released under the MIT License; see [`LICENSE`](LICENSE).

The repository also bundles third-party code, each under its own license,
with a `LICENSE` file in the corresponding directory:

| Component | Path | Origin | License |
|---|---|---|---|
| FastGRNN inference | `firmware/fastgrnn/` | [Microsoft EdgeML](https://github.com/microsoft/EdgeML) | MIT |
| CMSIS-NN kernels | `firmware/cmsis_nn/` | [Arm CMSIS-NN](https://github.com/ARM-software/CMSIS-NN) | Apache-2.0 |
| SimpleSerial + build glue | `firmware/simpleserial/`, `firmware/Makefile.inc` | [ChipWhisperer](https://github.com/newaetech/chipwhisperer) (NewAE) | Apache-2.0 |
| SAM4S HAL | `firmware/hal/` | Atmel/Microchip ASF | Microchip ASF license |

## Citation

If you use this code or build on this work, please cite the [paper](https://ieeexplore.ieee.org/document/11604834):

```bibtex
@inproceedings{11604834,
  author    = {Etim, Anthony and Nampally, Srilalith and Rasouli, Aubtin and
               Mazza, Dustin and Chilakapati, Krishna and Chiu, Tinghung and
               Erata, Ferhat and Nazhandali, Leyla and Xiong, Wenjie and Szefer, Jakub},
  booktitle = {2026 IEEE International Symposium on Hardware Oriented Security and Trust (HOST)},
  title     = {Fault Injection Attacks and Countermeasures on TinyML Algorithms},
  year      = {2026},
  pages     = {68--78},
  doi       = {10.1109/HOST68814.2026.11604834},
  url       = {https://doi.ieeecomputersociety.org/10.1109/HOST68814.2026.11604834},
  publisher = {IEEE Computer Society},
  address   = {Los Alamitos, CA, USA},
  month     = may
}
```
