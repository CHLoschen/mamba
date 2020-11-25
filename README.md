# mamba
A python tool for the perception of chemical bonds via machine learning.  
See also:  
[Perception of Chemical Bonds via Machine Learning](https://chemrxiv.org/articles/preprint/Perception_of_Chemical_Bonds_via_Machine_Learning/7403630/2)

Internally mamba uses RDKit, numpy, pandas and scikit-learn python packages

For help run:
mamba.py -h

Examples:
    
1) Train with large SD file:  
  mamba.py --train largefile.sdf  
  
2) [Optional] Add some SD file:  
  mamba.py --add special_case.sdf  
  
3) Predict bonds for xyz or pdb file:  
  mamba.py --predict new_molecule.xyz  
  
[unzip .sdf files to obtain some training files]

## Installation

For installation a few python libraries are necessary, install them via pip:

`pip install -r requirements.txt`

RDKit affords conda for installation:

`conda install -c conda-forge rdkit
`

