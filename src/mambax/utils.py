"""Utility functions for data processing and visualization."""

import logging
import random
import sys
from typing import Generator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit.Chem import AllChem as Chem
from sklearn.metrics import classification_report, confusion_matrix

try:
    import seaborn as sns
except ImportError:
    sns = None


def set_logging(loglevel: int = logging.INFO) -> None:
    """Configure logging to file and stdout."""
    logging.basicConfig(filename='log.txt', level=loglevel, format='%(asctime)s - %(levelname)s - %(message)s')
    root = logging.getLogger()
    root.setLevel(loglevel)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(loglevel)
    formatter = logging.Formatter('%(message)s')
    ch.setFormatter(formatter)
    root.addHandler(ch)


def mol_dataframe_generator(X: pd.DataFrame) -> Generator:
    """Generator that groups dataframe by molecule ID."""
    X['index_mod'] = X.index.str.split("_pos")
    X['index_mod'] = X.index_mod.str[1]
    X['index_mod'] = X.index_mod.str.split("_")
    X['index_mod'] = X.index_mod.str[0]
    grouped = X.groupby('index_mod')
    for name, df_sub in grouped:
        yield name, df_sub


def plot_classification_results(y: np.ndarray, ypred: np.ndarray) -> None:
    """Plot classification results as confusion matrix."""
    report = classification_report(y, ypred, digits=3, target_names=['X', '-', '=', '#', 'a'])
    print(report)

    cm = confusion_matrix(y, ypred, labels=[0, 1, 2, 3, 4])
    df_cm = pd.DataFrame(cm, index=[i for i in "X-=#a"], columns=[i for i in "X-=#a"])
    plt.figure(figsize=(10, 7))
    if sns is not None:
        sns.set(font_scale=1.4)
        ax = sns.heatmap(df_cm, annot=True, fmt='d', annot_kws={"size": 16}, cmap="magma_r")
    else:
        ax = plt.gca()
        im = ax.imshow(df_cm, cmap="magma_r")
        plt.colorbar(im)

    ax.set(xlabel='predicted bond type', ylabel='true bond type')
    plt.show()


def shuffle_split_sdfile(infile: str = 'test.sdf', frac: float = 0.8, rseed: int = 42) -> None:
    """Shuffle and split SDF file into train/test sets."""
    random.seed(rseed)
    trainfile = infile.replace('.sdf', '_train.sdf')
    w1 = Chem.SDWriter(trainfile)
    w1c = 0
    testfile = infile.replace('.sdf', '_test.sdf')
    w2 = Chem.SDWriter(testfile)
    w2c = 0
    suppl = Chem.SDMolSupplier(infile, removeHs=False, sanitize=False)
    for i, mol in enumerate(suppl):
        rdm = random.random()
        if rdm < frac:
            w1.write(mol)
            w1c += 1
        else:
            w2.write(mol)
            w2c += 1

    w1.close()
    w2.close()
    print("Finished writing %d train moles / %d test moles" % (w1c, w2c))


def sanitize_sdf(infile: str = 'test.sdf', removeHs: bool = False) -> None:
    """Sanitize SDF file using RDKit."""
    outfile = infile.replace('.sdf', '_sane.sdf')
    w = Chem.SDWriter(outfile)
    suppl = Chem.SDMolSupplier(infile, removeHs=removeHs, sanitize=True)
    count = 0
    for i, mol in enumerate(suppl):
        if mol is not None:
            w.write(mol)
        else:
            count += 1

    logging.info("Removed %d files - saving .sdf to: %s " % (count, outfile))
    w.close()


def clean_sdf(infile: str = 'test.sdf') -> None:
    """Clean SDF file by removing unreasonable structures."""
    from .features import extract_features

    outfile = infile.replace('.sdf', '_clean.sdf')
    w = Chem.SDWriter(outfile)
    suppl = Chem.SDMolSupplier(infile, removeHs=False, sanitize=False)
    count = 0
    for i, mol in enumerate(suppl):
        df = extract_features(mol, infile, i, verbose=False, printHeader=True, useSelectionRules=True)
        if df is not None and mol is not None:
            w.write(mol)
        else:
            count += 1

    logging.info("Removed %d files - saving .sdf to: %s " % (count, outfile))
    w.close()


def get_stats(infile: str, skipH: bool = False, addBonds: bool = False) -> None:
    """Generate statistical analysis of SDF file."""
    from .features import convert_sdf2dataframe

    import matplotlib.pyplot as plt

    df = convert_sdf2dataframe(infile=infile, outfile=None, fillNa=9999.0, verbose=False, skipH=skipH, addBonds=addBonds)
    df = df[df.bond > 0]
    bonds = [['C', 'C'], ['O', 'C'], ['N', 'C'], ['P', 'C'], ['S', 'C'], ['P', 'O'], ['S', 'O']]
    n_rows = 2
    n_cols = 4

    pt = Chem.GetPeriodicTable()
    fig = plt.figure()
    plt.suptitle(infile)
    for i, (a, b) in enumerate(bonds):
        title = str(a) + '&' + str(b)
        a_num = pt.GetAtomicNumber(a)
        b_num = pt.GetAtomicNumber(b)

        idx = ((df.ata.values == a_num) & (df.atb.values == b_num)) | ((df.ata.values == b_num) & (df.atb.values == a_num))
        df_tmp = df[idx]

        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        ax.set_title(title)
        ax.set_xlabel('dist')
        data = [df_tmp.distab[df_tmp.bond == 1], df_tmp.distab[df_tmp.bond == 2],
                df_tmp.distab[df_tmp.bond == 3], df_tmp.distab[df_tmp.bond == 4]]
        labels = ['-', '=', '#', 'a']

        ax.hist(data, bins=20, alpha=0.5, stacked=True, histtype='bar', label=labels)
        fig.legend(labels=labels, loc='upper right', ncol=5, labelspacing=0.)
    fig.tight_layout()
    plt.show()

