# mambax

**MAMBAX - Machine Learning Meets Bond Analytix**

A python tool for the perception of chemical bonds via machine learning.
See also:
[Perception of Chemical Bonds via Machine Learning](https://chemrxiv.org/articles/preprint/Perception_of_Chemical_Bonds_via_Machine_Learning/7403630/2)

Internally mambax uses RDKit, numpy, pandas and scikit-learn python packages

For help run:

```bash
mambax -h
```

Examples:

1) Train with large SD file:

```bash
  mambax --train largefile.sdf
```

2) [Optional] Add some SD file:

```bash
  mambax --add special_case.sdf
```

3) Create SDF with bond information from xyz or pdb:

```bash
  mambax --predict new_molecule.xyz
```

## Installation

Install from source:

```bash
git clone https://github.com/CHLoschen/mamba.git
cd mamba
pip install -e .
```

You can also use mambax as a Python package:

```python
from mambax import extract_features, predict_bonds, train_job

# Use the functions programmatically
```
