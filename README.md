# Multi-scale Learning with Sparse Residual Network for Explanable Multi-disease Diagnosis in OCT Images
by Phuoc-Nguyen Bui, Duc-Tai Le, Junghyun Bum, Seongho Kim, Su Jeong Song, Hyunseung Choo.

## Introduction
This repository is for our paper in Bioengineering journal [Multi-scale Learning with Sparse Residual Network for Explanable Multi-disease Diagnosis in OCT Images](https://doi.org/10.3390/bioengineering10111249)

## Installation
This repository is implemented in PyTorch 1.13

## Usage
Step 1. Clone the repository

```
git clone https://github.com/phuocnguyen2008/MSL-SRN
cd MSL-SRN
```

Step 2. Put the data in ``./data/``

Step 3. Run the training command

``
python train.py
``

Step 4. Run the inference command

``
python validate_doublenet.py
``

## Citation
If you find this MSL-SRN helpful for your research, please consider citating it:
```
@article{bui2023multi,
  title={Multi-Scale Learning with Sparse Residual Network for Explainable Multi-Disease Diagnosis in OCT Images},
  author={Bui, Phuoc-Nguyen and Le, Duc-Tai and Bum, Junghyun and Kim, Seongho and Song, Su Jeong and Choo, Hyunseung},
  journal={Bioengineering},
  volume={10},
  number={11},
  pages={1249},
  year={2023},
  publisher={MDPI}
}
```
