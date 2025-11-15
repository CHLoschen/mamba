"""Backward compatibility module - re-exports all functions from modular structure."""

# Re-export all functions for backward compatibility
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

__all__ = [
    # Features
    "extract_features",
    "convert_sdf2dataframe",
    "convert_sdfiles2csv",
    # ML
    "train_from_csv",
    "train_job",
    "predict_bonds",
    "generate_predictions",
    "generate_multi_predictions",
    # Evaluation
    "eval_job",
    "evaluate",
    "evaluate_OB",
    # IO
    "read_pdbfile",
    "read_mol",
    "read_xyz",
    "create_sdfile",
    "mol2xyz",
    "convert_smiles2sdfile",
    # Utils
    "shuffle_split_sdfile",
    "get_stats",
    "clean_sdf",
    "sanitize_sdf",
    "mol_dataframe_generator",
    "plot_classification_results",
    "set_logging",
]
