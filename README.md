# DCRLA

## Pinch Force Evaluation and DCRLA Model Training

This repository provides the de-identified statistical data, derived feature
matrices, preprocessing definitions, feature-extraction code, and model-training
code used to evaluate sustained and explosive pinch force for stroke screening.
The proposed model is DCRLA, which combines DPMS-CNN feature extraction,
random-forest feature ranking, nested feature-count selection, demographic
fusion, an LSTM, and an attention mechanism.

## Repository Structure

### Data and Derived Features

- `32.xlsx`: de-identified statistical data for the 32-subject explosive-pinch cohort.
- `42.xlsx`: de-identified statistical data for the sustained-pinch cohort. The analysis reported in the manuscript uses 40 eligible participants.
- `fused_features_32sub_3s_60ch.npy`: 60-channel DPMS-CNN feature matrix for the 3 s explosive-pinch task.
- `fused_features_60ch.npy`: 60-channel DPMS-CNN feature matrix for the 10 s sustained-pinch task.
- `Experiment20260721.rar`: raw measurement records and photographic evidence from the pinch-force acquisition system validation performed by five experimenters. The archive includes the loading setup, AIGU NK-100 reference readings, displayed system readings, and indoor temperature and relative-humidity records. It is available from [Baidu Netdisk](https://pan.baidu.com/s/1n7qFFSUtuZmww3TbDCV6Pg) using access code `we5v`.

The `.npy` files contain derived feature representations and do not contain the
original participant-level force recordings. Regenerate these files whenever
the preprocessing or DPMS-CNN configuration is changed.

### Feature Extraction

- `cnn-feature-morechannel_ver1.py`: 10 s sustained-pinch preprocessing and DPMS-CNN feature extraction.
- `cnn_feature_3s_32sub_ver1.py`: 3 s explosive-pinch DPMS-CNN feature extraction.

Both tasks retain the complete protocol-defined trial at 30 Hz. The fixed input
lengths are 300 samples for sustained pinch and 90 samples for explosive pinch;
longer sequences are truncated and shorter sequences are zero-padded.

### Model Training

- `40DCRLA_ver1.py`: DCRLA training and nested evaluation for sustained pinch.
- `32ownwithage_ver1.py`: DCRLA training and nested evaluation for explosive pinch.
- `40baselinewithage_ver1.py`: LSTM, Transformer, and DCLA baselines for sustained pinch.
- `32baselinewithage_ver1.py`: LSTM, Transformer, and DCLA baselines for explosive pinch.

The standard LSTM and Transformer baselines use the standardized single-channel
force sequence without demographic fusion or RF feature selection. DCLA uses
all 60 DPMS-CNN channels and demographic data but omits RF ranking and dynamic
feature-count selection. DCRLA performs RF channel ranking and selects the
number of retained channels within the inner cross-validation loop.

## DPMS-CNN Kernel Settings

The DPMS-CNN follows the deep and shallow parallel paths shown in Fig. 6 of the
manuscript. The shallow path contains two Conv1d layers followed by max pooling.
Its convolutional kernel lengths are 20 and 5 samples, with channel widths
`1 -> 32 -> 64`. The deep path contains a Conv1d layer with a 20-sample kernel,
max pooling, and three additional Conv1d layers with 5-sample kernels; its
channel widths are `1 -> 32 -> 64 -> 64 -> 64`. All convolutions use stride 1,
and padding preserves the intended temporal alignment. Max pooling uses a
kernel size and stride of 2. The deep and shallow outputs are concatenated and
processed by two fusion convolutions with 3-sample kernels and channel widths
`128 -> 128 -> 60`.

At 30 Hz, kernel lengths of 20, 5, and 3 samples correspond to temporal spans
of approximately 0.67 s, 0.17 s, and 0.10 s, respectively. The same kernel
configuration is used for both pinch tasks; only the fixed input duration differs.

## Training and Evaluation Configuration

The following settings correspond to Table II and the model-training description
in the manuscript.

| Model | Optimizer | Learning rate | Batch size | Evaluation / epochs |
|---|---|---:|---:|---|
| LSTM | Adam | 1 x 10^-4 | 8 | 10 folds / 150 epochs |
| Transformer | Adam | 1 x 10^-4 | 8 | 10 folds / 150 epochs |
| DCLA | Adam | 1 x 10^-4 | 8 | 10 folds / 150 epochs |
| DCRLA | Adam | 1 x 10^-4 | 8 | 10-fold outer, 5-fold inner / 150 epochs |

Shared training settings are:

- Adam weight decay: `5 x 10^-3`.
- Loss: weighted cross-entropy with weights calculated from the current training split.
- Dropout: `0.2`.
- Early stopping: validation loss with a patience of 15 epochs.
- DCRLA candidate feature counts: `{10, 15, 20, 25, 30, 40, 50, 60}`.
- LSTM hidden size: 64.
- Attention size: 32.
- Random seed: 42.
- Outer test folds are not used for training, early stopping, feature ranking, or hyperparameter selection.

Baseline-specific settings are:

- LSTM: one LSTM layer with 64 hidden units, temporal mean pooling, and a two-layer classifier.
- Transformer: input projection to `d_model=16`, one encoder layer, two attention heads, feed-forward dimension 32, and dropout 0.2.
- DCLA: all 60 DPMS-CNN channels, one 64-unit LSTM layer, attention size 32, demographic fusion, and no RF channel selection.
- DCRLA: RF channel ranking within the nested evaluation, a 64-unit LSTM, attention size 32, and demographic-guided attention.

## Data Privacy and Ethics

The original participant-level pinch-force time series and identifiable or
restricted clinical metadata are not publicly hosted because they contain
sensitive clinical information and are governed by the approved ethics and
data-protection procedures.

The public repository contains only source code, configuration information,
de-identified statistical materials approved for sharing, and abstracted
feature representations. Researchers who require access to the original raw
recordings for validation must submit a formal data-access request to the
corresponding author. The request should state the applicant's name,
institutional affiliation, and research purpose. Access remains subject to
institutional ethics approval, verification of the proposed use, and execution
of an applicable Data Use Agreement.

## Getting Started

Install Python and the required packages:

```bash
pip install numpy pandas torch scikit-learn matplotlib seaborn openpyxl
```

Update the input and output paths in the configuration block of each script,
generate or obtain the required derived feature files, and then run the
task-specific DCRLA or baseline script. Access to restricted raw force sequences
must be approved before the raw-sequence baseline or feature-generation pipeline
can be reproduced from participant-level recordings.
