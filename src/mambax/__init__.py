"""MAMBAX - Machine Learning Meets Bond Analytix

A python tool for the perception of chemical bonds via machine learning.
"""

# Import from modular structure
from .evaluation import eval_job, evaluate, evaluate_OB
from .features import convert_sdf2dataframe, convert_sdfiles2csv, extract_features
from .io import (
    convert_smiles2sdfile,
    create_sdfile,
    mol2xyz,
    read_mol,
    read_pdbfile,
    read_xyz,
)
from .ml import (
    generate_multi_predictions,
    generate_predictions,
    predict_bonds,
    train_from_csv,
    train_job,
)
from .utils import (
    clean_sdf,
    get_stats,
    mol_dataframe_generator,
    plot_classification_results,
    sanitize_sdf,
    set_logging,
    shuffle_split_sdfile,
)

__version__ = "0.1.0"
__author__ = "Christoph Loschen"

__all__ = [
    "extract_features",
    "convert_sdf2dataframe",
    "convert_sdfiles2csv",
    "train_from_csv",
    "train_job",
    "eval_job",
    "evaluate",
    "evaluate_OB",
    "generate_multi_predictions",
    "generate_predictions",
    "predict_bonds",
    "convert_smiles2sdfile",
    "shuffle_split_sdfile",
    "get_stats",
    "clean_sdf",
    "sanitize_sdf",
    "read_pdbfile",
    "read_mol",
    "read_xyz",
    "create_sdfile",
    "mol_dataframe_generator",
    "mol2xyz",
    "plot_classification_results",
    "set_logging",
]
