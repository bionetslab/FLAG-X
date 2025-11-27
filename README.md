
![Tests](badges/test-status.svg) ![Coverage](badges/coverage.svg) ![Python Version](https://img.shields.io/badge/python-%3E%3D3.10%20%3C3.14-blue?logo=python)


# FLow cytometry Automated Gating - toolboX (FLAG-X)

FLAG-X is a Python toolbox for **automated, end-to-end cytometry data processing**, including:
- Data loading (FCS/CSV → AnnData)
- Alignment of channel names arcoss samples 
- Sample-wise preprocessing
- Train/(val)/test splitting on the sample level
- Sample-wise downsampling of training data
- Model Training MLP (supervised), SOM (supervised or unsupervised)
- Model saving
- Inference on new data:
  - Dimensionality reduction (SOM, UMAP, t-SNE, PCA, etc.)
  - Automated gating (cell type prediction)
  - Export of annotated samples to FCS format for downstream analysis using standard flow-cytometry tools.

FLAG-X provides a streamlined pipeline and a command line interface (CLI) for users with little programming experience.


## Installation
From source using **conda** or **mamba**:
```console
git clone git@github.com:bionetslab/FLAG-X.git
cd FLAG-X
mamba env create -f environment.yml
pip install -e .
```

From source using **pixi**:
```console
git clone git@github.com:bionetslab/FLAG-X.git
cd FLAG-X
pixi install
```

**NOTE:** The environments provided in this project install the CPU-only version of PyTorch. 
Users who require GPU acceleration must install a CUDA-enabled PyTorch build themselves following the instructions at [PyTorch get started](https://pytorch.org/get-started/locally/).


## To do
- [ ] Add functionality to apply compensation
- [ ] Add functionality to load lmd files


## License

This project is licensed under the **GNU General Public License v3.0** - see the [LICENSE](LICENSE) file for details.

