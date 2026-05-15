# DCRLA
# Pinch Force Evaluation and DCRLA Model Training

This repository contains the de-identified statistical data, fused feature files, and training scripts for our research on **Explosive Pinch Force** and **Sustained Pinch Force**. Our work introduces the **DCRLA model** and evaluates it against several baseline architectures using multi-level fused features.

## 📂 Repository Structure

### 📊 Data & Features
* **`32.xlsx`** De-identified statistical data for the **Explosive Pinch Force** experiments.
* **`42.xlsx`** De-identified statistical data for the **Sustained Pinch Force** experiments.
* **`fused_features_32sub_3s_60ch.npy`** Feature matrix for explosive pinch force, extracted using a fusion of deep and shallow features (32 subjects, 60 channels).
* **`fused_features_60ch.npy`** Feature matrix for sustained pinch force, extracted using a fusion of deep and shallow features (60 channels).

### 💻 Training Source Code
* **`Baselinewithage.py`** Implementation and training scripts for three baseline models used for performance comparison.
* **`DCRLAwithage.py`** Training script for the proposed **DCRLA (Deep Contextual Representation Learning Architecture)** model.

---

## 🔒 Data Privacy & Ethics Statement

Due to the sensitive nature of clinical patient data and medical ethics requirements, the raw time-series pinch force data and identifiable clinical information are not publicly hosted in this repository. 

**Access to Raw Data:**
The files provided (`.xlsx` and `.npy`) are either fully de-identified statistical summaries or abstracted feature representations. To obtain access to the **original raw data** for research validation purposes:
1. Please submit a formal data access request to: `[Insert Your Email Address]`
2. The request should include your name, institutional affiliation, and a brief description of your research purpose.
3. Data sharing is subject to the signing of a Data Use Agreement (DUA) to ensure patient confidentiality.

---

## 🚀 Getting Started

### Prerequisites
Ensure you have Python installed along with the following dependencies:
```bash
pip install numpy pandas torch scikit-learn
