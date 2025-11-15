"""Evaluation functions for model performance assessment."""

import logging
import subprocess
import sys
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Draw
from sklearn.metrics import accuracy_score, f1_score

try:
    import seaborn as sns
except ImportError:
    sns = None

from .features import extract_features
from .io import mol2xyz
from .ml import generate_predictions
from .utils import mol_dataframe_generator, plot_classification_results


def evaluate(
    filename_test: str,
    filename_train: str = 'train_dat.csv',
    plotting: bool = True,
    iterative: bool = False,
    verbose: bool = False
) -> pd.DataFrame:
    """Evaluate classifier on test dataset."""
    df = pd.read_csv(filename_test, index_col=0)
    if filename_train is not None:
        logging.info("Analyze train data...")
        df_train = pd.read_csv(filename_train,index_col=0)
        print(df_train.shape)
        df_train['bondtype']=df_train['bond'].astype('category')

        df_train = df_train[df_train.ata==6]
        df_train = df_train[df_train.atb==6]
        if plotting:
            if sns is not None:
                ax = sns.boxplot(x="bond", y="distab", data=df_train[['distab', 'bond']])
                ax.set(ylabel='C-C distance', xlabel='bond type')
                plt.show()

    logging.info("Evaluate data set: " + filename_test)
    logging.info("Loading classifier...")
    import pickle
    with open("clf.p", "rb") as f:
        clf = pickle.load(f)

    logging.info("Loading test set with %d rows from file %s\n"%(df.shape[0],filename_test))

    y = df['bond']
    X = df.drop(['bond','id1','id2'],axis=1,inplace=False)

    yprob = clf.predict_proba(X)
    ypred = clf.predict(X)

    score = accuracy_score(y,ypred)
    score2 = f1_score(y,ypred,average='weighted')

    logging.info("ACCURACY:%0.3f - F1-score: %0.3f\n" % (score,score2))

    X['bond_pred'] = ypred
    X['p(-)'] = yprob[:, 1]
    X['p(=)'] = yprob[:, 2]
    X['p(#)'] = yprob[:, 3]
    X['p(a)'] = yprob[:, 4]
    X['bond'] = y

    if plotting:
        print("Misclassification stats:")
        idx = (ypred != y)
        df_tmp = X[idx.values]
        print(df_tmp[['ata','atb','distab','bond','bond_pred']].head(200).sort_values(['ata']))

        plot_classification_results(y,ypred)


    mol_df_list = mol_dataframe_generator(X)
    all=0
    ok=0
    not_ok=0
    false_indices=[]
    for name, df_sub in mol_df_list:
        all += 1
        if iterative:
            print("ERROR: Iterative - does not work in fast evaluation mode..")
            sys.exit(1)

        if np.array_equal(df_sub['bond_pred'].values, df_sub['bond'].values):
            ok += 1
        else:
            not_ok += 1
            mask = df_sub['bond_pred'] != df_sub['bond']
            idx = np.argmax(mask)
            false_indices.append(idx)

    acc = ok/float(all)
    print(false_indices)
    print("\nTOTAL: %5d OK: %5d WRONG: %5d Accuray: %6.3f"%(all,ok,not_ok,acc))

    return(X)


def evaluate_OB(filename: str = 'fullerene_ml.sdf', verbose: bool = False) -> None:
    """Evaluate predictions via OpenBabel comparison."""
    logging.info("Evaluating %s via OBabel" % filename)

    suppl = Chem.SDMolSupplier(filename, removeHs=False, sanitize=True)
    nok = 0
    nfalse = 0
    nall = len(suppl)
    for i, mol in enumerate(suppl):
        if mol is None: continue
        xyz_str = mol2xyz(mol)
        #remove H for comparison with OB
        mol = Chem.RemoveHs(mol)
        df_orig = extract_features(mol, "babel_orig", (i+1), skipH=True)
        if df_orig is None: continue
        bond_orig = df_orig['bond']
        cmd_call = ["obabel", "-ixyz", "-osdf"]
        p = subprocess.Popen(cmd_call, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        molblock, err = p.communicate(xyz_str.encode())
        mol_pred = Chem.MolFromMolBlock(molblock.decode(), removeHs=True, sanitize=False)

        if mol_pred is None:
            nfalse += 1
            continue
        df = extract_features(mol_pred, "obabel", 0, skipH=True)
        if df is None:
            nfalse += 1
            continue
        if len(bond_orig)!=len(df['bond'].values):
            logging.error("Original (%d) and predicted bond vector (%d) have different length!"%(len(bond_orig),len(df['bond'].values)))
            if verbose:
                mol_pred_noH = Chem.RemoveHs(mol_pred)
                Chem.Compute2DCoords(mol_pred_noH)
                Chem.Compute2DCoords(mol)
                img = Draw.MolsToGridImage([mol_pred_noH, mol], molsPerRow=2, subImgSize=(400, 400),
                                           legends=['ob' + str(i + 1), 'orig' + str(i + 1)])
                img.show()

        if np.array_equal(bond_orig.values, df['bond'].values):
            nok+=1
        else:
            if verbose:
                mol_pred_noH = Chem.RemoveHs(mol_pred)
                Chem.Compute2DCoords(mol_pred_noH)
                Chem.Compute2DCoords(mol)
                img = Draw.MolsToGridImage([mol_pred_noH, mol], molsPerRow=2, subImgSize=(400, 400),
                                           legends=['ob' + str(i + 1), 'orig' + str(i + 1)])
                img.show()
                res = input()
                if 'n' in res.lower() or "f" in res.lower():
                    nfalse += 1
                    print("FALSE: %d/%d" % (nfalse, len(suppl)))
                else:
                    nok += 1
                    print("OK: %d/%d" % (nok,len(suppl)))
                if verbose:
                    with open('ob_failure'+str(i+1)+'.sdf', 'w') as f:
                        f.write(molblock.decode())
                    with open('ob_reference'+str(i+1)+'.sdf', 'w') as f:
                        f.write(Chem.MolToMolBlock(mol))
            else:
                idx = np.where(bond_orig.values != df['bond'].values)
                for k in idx[0]:
                    try:
                        ata = df_orig.iloc[k][['ata']].values[0]
                        atb = df_orig.iloc[k][['atb']].values[0]
                    except IndexError:
                        print(Chem.MolToSmiles(mol))
                        continue

                    if (int(ata)==8 and int(atb)== 7) or ( int(ata)==8 and int(atb)== 6) or ( int(ata)==15 and int(atb)== 8):
                        bond_orig.values[k]=1
                        df['bond'].values[k]=1

                if np.array_equal(bond_orig.values, df['bond'].values):
                    nok += 1
                    print("OK: %d/%d" % (nok, len(suppl)))
                else:
                    nfalse += 1
                    with open('ob_failure'+str(i+1)+'.sdf', 'w') as f:
                        f.write(molblock.decode())
                    with open('ob_reference'+str(i+1)+'.sdf', 'w') as f:
                        f.write(Chem.MolToMolBlock(mol))
                    print("FALSE: %d/%d (%6.3f%%)" % (nfalse, len(suppl), 1.0-nfalse/float(i+1)))

    acc = nok / float(nall)
    logging.info("\nTOTAL: %5d OK: %5d WRONG: %5d Accuray: %6.3f" % (nall, nok, nfalse, acc))


def eval_job(
    filename: str,
    skipH: bool = False,
    iterative: bool = False,
    verbose: bool = False
) -> None:
    """Evaluate predictions per molecule."""
    import pickle

    print("Evaluation run with option: noH(%r)" % (skipH))
    print("Loading classifier...")
    with open('clf.p', "rb") as f:
        clf = pickle.load(f)
    if iterative:
        with open('clf_iter.p', "rb") as f:
            clf_iter = pickle.load(f)
    else:
        clf_iter = None
    suppl = Chem.SDMolSupplier(filename, removeHs=skipH, sanitize=iterative)
    nok = 0
    nfalse = 0
    for i, mol in enumerate(suppl):
        if mol is None: continue
        res = generate_predictions(mol, skipH=skipH, iterative=True, forceAromatics=False, maxiter=1, verbose=verbose,
                                   clf=clf, clf_iter=clf_iter, isEval=True)
        if res is None: continue
        if i % 50 == 0:
            logging.info("%d %r\n" % (i, res))
        if res:
            nok += 1
        else:
            nfalse += 1

    nall = len(suppl)
    acc = nok / float(nall)
    logging.info("\nTOTAL: %5d OK: %5d WRONG: %5d Accuray: %6.3f" % (nall, nok, nfalse, acc))

