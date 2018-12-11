# mamba
A python tool for the perception of chemical bonds via machine learning

Internally uses RDKit, numpy, pandas and scikit-learn python packages

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

