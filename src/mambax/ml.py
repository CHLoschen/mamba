"""Machine learning training and prediction functions."""

import logging
import os
import pickle
import sys
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from rdkit.Chem import AllChem as Chem
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from .features import convert_sdfiles2csv, extract_features
from .io import create_sdfile, read_mol, read_pdbfile, read_xyz


def train_from_csv(
    filename: str,
    grid_search: bool = False,
    useRF: bool = False,
    plotClassifier: bool = False,
    save_clf: str = 'clf.p',
    verbose: bool = False  # Unused but kept for API compatibility
) -> Any:
    """Train classifier from CSV file and save model."""
    logging.info("Training data on dataset:")
    df = pd.read_csv(filename,index_col=0)
    if 'id1' in df.columns and 'id2' in df.columns:
        df.drop(['id1', 'id2'], axis=1,inplace=True)
    logging.info("Shape   : %d X %d" % (df.shape[0], df.shape[1]))
    logging.info("Features: %s" % (df.columns))
    logging.info("Dropping duplicates...")
    df.drop_duplicates(inplace=True)
    logging.info("Shape   : %d X %d" % (df.shape[0], df.shape[1]))

    y = df['bond']
    X = df.drop(['bond'],axis=1,inplace=False)

    if plotClassifier:
        tree = DecisionTreeClassifier( max_depth=5)
        tree.fit(X,y)
        dot_data = tree.export_graphviz(tree, out_file='tree')
        import graphviz
        graph = graphviz.Source(dot_data)
        graph.render("decisiontree")

    n_jobs = 1
    n_splits = 4
    if useRF:
        model = RandomForestClassifier(n_estimators=250, max_depth=None, min_samples_leaf=5, n_jobs=n_jobs,
                                       max_features=11, oob_score=False)
    else:
        model = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.1, max_depth=5, verbose=1)
        parameters = {}


    if grid_search:
        n_jobs = 4
        cv = StratifiedKFold(n_splits=n_splits)
        model = GridSearchCV(model, parameters, n_jobs=n_jobs, verbose=2, scoring='f1_micro', cv=cv,refit=True)
        model.fit(X,y)

        means = model.cv_results_['mean_test_score']
        stds = model.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, model.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print(model)
    else:
        logging.info("Fitting classifier: %s"%(model))
        model.fit(X, y)
    with open(save_clf, "wb") as f:
        pickle.dump(model, f)
    logging.info("Saving classifier as: %s"%(save_clf))
    return model


def train_job(
    filename: str,
    reset: bool = True,
    eval: bool = False,
    fmethod: str = 'UFF',
    skipH: bool = False,
    iterative: bool = False,
    sample: Optional[float] = None,
    useRF: bool = False,
    verbose: bool = False
) -> None:
    """Train classifier from SDF or SMILES file."""
    from .evaluation import evaluate

    if eval:
        train_file = 'eval_dat.csv'
        reset=True
    else:
        train_file = 'train_dat.csv'

    iter_file = ""
    if iterative and not eval:
        logging.info("Iterative mode switched ON!")
        iter_file = train_file.replace("_dat","_iter")

    if useRF and not eval:
        logging.info("INFO: Using Random Forest for training!")

    if reset:
        if os.path.isfile(train_file):
            os.remove(train_file)
        if os.path.isfile(iter_file):
            os.remove(iter_file)

    if filename.endswith('.sdf') or filename.endswith('.sd'):
        convert_sdfiles2csv(file_list=[filename], outdat=train_file, skipH=skipH, addBonds=False, sample=sample, verbose=verbose)
        if iterative and not eval:
            convert_sdfiles2csv(file_list=[filename], outdat=iter_file, skipH=skipH, addBonds=True, sample=sample, verbose=verbose)

    elif filename.endswith('.smi'):
        logging.info("Using forcefield for optimization: %s" % (fmethod))
        convert_sdfiles2csv(file_list=[filename], outdat=train_file, method=fmethod, skipH=skipH, addBonds=False)
        if iterative and not eval:
            convert_sdfiles2csv(file_list=[filename], outdat=iter_file, method=fmethod, skipH=skipH, addBonds=True, verbose=verbose)

    if not os.path.isfile(train_file):
        sys.stderr.write("ERROR: Missing training data file: %s!\n"%(train_file))
        sys.exit(1)

    if eval:
        evaluate(train_file,iterative=iterative, verbose=verbose)

    else:
        train_from_csv(train_file, useRF=useRF, verbose=verbose)
        if iterative:
            train_from_csv(iter_file,useRF=useRF, save_clf="clf_iter.p", verbose=verbose)


def predict_bonds(
    filename: Union[str, Any],
    q: int = 0,
    skipH: bool = False,
    iterative: bool = False,
    forceAromatics: bool = False,
    clf: Optional[Any] = None,
    eval: bool = False,
    verbose: bool = False
) -> Optional[pd.DataFrame]:
    """Predict bonds for a molecule from SDF/mol or XYZ file."""

    if not iterative:
        if eval:
            m = filename
            df = extract_features(m, "rdkitmol", 0, verbose=verbose, printHeader=verbose, fillNa=9999.0, xyz_file=None,
                                  skipH=skipH, addBonds=False, useSelectionRules=False)
            if df is None:
                return None
            bond_orig = df['bond']
        else:
            df = extract_features(None, filename, 0, verbose=False, printHeader=False, fillNa=9999.0, xyz_file=filename,
                                  skipH=skipH)
            if df is None:
                logging.info("Could not generate dataframe from %s" % filename)
                return None
        clf_name = "clf.p"
    else:
        m = Chem.MolFromMolBlock(filename, sanitize=False)
        df = extract_features(m, "rdkitmol", 0, verbose=verbose, printHeader=verbose, fillNa=9999.0, xyz_file=None,
                              skipH=skipH, addBonds=True, useSelectionRules=False)
        if df is None:
            return None
        df['q'] = q
        clf_name = "clf_iter.p"

    order = df[['id1', 'id2']]
    X = df.drop(['bond', 'id1', 'id2'], axis=1)
    if verbose:
        print("X shape n=%d m=%d " % (X.shape[0], X.shape[1]))


    if clf is None:
        print("Loading classifier %s..." % (clf_name))
        with open(clf_name, "rb") as f:
            clf = pickle.load(f)
    yprob = clf.predict_proba(X)
    ypred = clf.predict(X)

    try:
        X['p(X)'] = yprob[:,0]
        X['p(-)'] = yprob[:,1]
        X['p(=)'] = yprob[:,2]
        X['p(#)'] = yprob[:,3]
        X['p(a)'] = yprob[:,4]
        X['S'] = 0.0
        for i in [0,1,2,3,4]:
            X['S'] = X['S'] - np.log(yprob[:,i]+1E-15)*yprob[:,i]
        X['bond'] = ypred
        X['id1'] = order['id1']
        X['id2'] = order['id2']

    except IndexError as e:
        sys.stderr.write(f"I/O error: {e}\n")
        sys.stderr.write("Are there enough training data for all bond types?\n")
        sys.exit(1)

    X = X[['id1', 'id2', 'q', 'ata', 'atb', 'distab', 'bond', 'p(-)', 'p(=)', 'p(#)', 'p(a)', 'p(X)', 'S']]

    # check high aromaticity case
    if forceAromatics:
        logging.info("Forcing aromaticity for bonds with p(a)>0.4!")
        potaroma_idx = X['p(a)'] > 0.4
        X.loc[potaroma_idx, 'bond'] = 4

    if eval:
        X['bond_orig'] = bond_orig

    if skipH:
        X = X[(X.ata != 1) | (X.atb == 1)]

    return(X)


def generate_predictions(
    filename: Union[str, Any],
    skipH: bool = False,
    iterative: bool = False,
    forceAromatics: bool = False,
    maxiter: int = 1,
    isEval: bool = False,
    writeFile: bool = False,
    verbose: bool = True,
    **kwargs: Any
) -> Union[str, bool, Tuple[bool, str], None]:
    """Generate predictions for single molecule."""
    clf = None
    clf_iter = None
    if kwargs and 'clf' in kwargs:
        clf = kwargs['clf']
        clf_iter = kwargs.get('clf_iter')

    if iterative and not os.path.isfile("clf_iter.p"):
        sys.stderr.write("ERROR: Please train classifier in iterative mode (--iterative) first!\n")
        sys.exit(1)
    elif iterative:
        if clf_iter is None:
            print("Loading classifier clf_iter.p...")
            with open('clf_iter.p', "rb") as f:
                clf_iter = pickle.load(f)

    if isinstance(filename, Chem.rdchem.Mol):
        mol = filename
        filename = 'eval'
        atomtypes, coords, q, _ = read_mol(mol)
        X = predict_bonds(mol, q=q, skipH=skipH, iterative=False, forceAromatics=forceAromatics, clf=clf, eval=True, verbose=False)
        if X is None:
            return None

    elif not filename.lower().endswith('.xyz') and not filename.lower().endswith('.pdb'):
        sys.stderr.write("ERROR: Need .xyz/.pdb file for prediction!\n")
        sys.exit(1)

    elif not os.path.isfile("clf.p"):
        sys.stderr.write("ERROR: Please train classifier first!\n")
        sys.exit(1)

    elif filename.lower().endswith(".xyz"):
        atomtypes, coords, q, _ = read_xyz(filename, skipH=skipH)
        X = predict_bonds(filename, q=q, skipH=skipH, iterative=False, forceAromatics=forceAromatics, clf=clf, verbose=verbose)

    elif filename.lower().endswith(".pdb"):
        atomtypes, coords, q, _ = read_pdbfile(filename, skipH=skipH)
        X  = predict_bonds(filename, q=q, skipH=skipH, iterative=False, forceAromatics=forceAromatics, clf=clf, verbose=verbose)

    if isEval:
        writeFile=False

    if verbose:
        print(X.head(10))
        print("Total Entropy: %6.2f" % (X['S'].sum()))
        print("Max Entropy  : %6.2f" % ( X['S'].max()))
        print("Bonds        : %6d" % ((X['bond']>0)).sum())

    ins = create_sdfile(filename, atomtypes, coords, X)

    if iterative:
        if verbose: print("Iterative prediction using bond estimates...")
        if isEval:
            bond_orig = X['bond_orig']
            X = X.drop(['bond_orig'], axis=1)

        maxiter = 10
        poscounter =0
        while poscounter<maxiter:
            X2 = predict_bonds(ins, q=q, skipH=skipH, iterative=True, forceAromatics=forceAromatics, clf=clf_iter, verbose=False)
            if X2 is None:
                return(False)
            if verbose:
                print(X2.head(40))

            dX2 = pd.DataFrame(index=X2.index)
            dX2['dp(-)'] = X2['p(-)'] - X['p(-)'].values
            dX2['dp(=)'] = X2['p(=)'] - X['p(=)'].values
            dX2['dp(#)'] = X2['p(#)'] - X['p(#)'].values
            dX2['dp(a)'] = X2['p(a)'] - X['p(a)'].values
            grad = dX2.values
            dX2['update'] = np.max(grad, axis=1) > 0.25
            mask = dX2['update'].values
            idx = np.where(mask)[0]
            if poscounter<len(idx):
                mask[idx[poscounter]] = False
            else:
                break

            if verbose:
                logging.info("Total Entropy: %6.2f" % (X2['S'].sum()))
                logging.info("Max Entropy  : %6.2f" % (X2['S'].max()))
                logging.info("Bonds        : %6d" % ((X2['bond'] > 0)).sum())
                logging.info("Max grad  : %6.2f" % (grad.max()))

            X.loc[mask,'bond'] = X2.loc[mask,'bond'].values
            X.loc[mask, 'p(-)'] = X2.loc[mask, 'p(-)'].values
            X.loc[mask, 'p(=)'] = X2.loc[mask, 'p(=)'].values
            X.loc[mask, 'p(#)'] = X2.loc[mask, 'p(#)'].values
            X.loc[mask, 'p(a)'] = X2.loc[mask, 'p(a)'].values

            if verbose: print(X.head(40))


            dX2['bond_X'] = X['bond'].values
            dX2['bond_X2'] = X2['bond'].values

            ins = create_sdfile(filename, atomtypes, coords, X)
            poscounter+=1

        if isEval:
            dX2['bond_orig'] = bond_orig
            if verbose:
                print(bond_orig)
                print(X['bond'])

            if np.array_equal(bond_orig.values, X['bond'].values):
                return True
            else:
                return False

    if writeFile:
        filename = os.path.basename(filename).replace('.xyz', '_ml.sdf').replace('.pdb', '_ml.sdf')
        with open(filename, 'w') as f:
            f.write(ins)

        logging.info("ML-generated SDF written to: " + filename)
    elif isEval:
        if np.array_equal(X.bond_orig.values, X['bond'].values):
            return (True, ins)
        else:
            if verbose:
                print(X[['bond', 'bond_orig']])
            return (False, ins)
    else:
        return ins


def generate_multi_predictions(
    filename_list: List[str],
    skipH: bool = False,
    iterative: bool = False,
    forceAromatics: bool = False,
    maxiter: int = 1,
    isEval: bool = False,
    verbose: bool = False,
    **kwargs: Any
) -> None:
    """Generate predictions for multiple XYZ files."""

    with open('clf.p', "rb") as f:
        clf = pickle.load(f)
    if iterative:
        with open('clf_iter.p', "rb") as f:
            clf_iter = pickle.load(f)
    else:
        clf_iter = None

    res_filename = 'multi.sdf'
    w = Chem.SDWriter(res_filename)
    w.SetKekulize(False)
    for f in filename_list:
        ins = generate_predictions(f, skipH=skipH, iterative=iterative, forceAromatics=forceAromatics, maxiter=maxiter, isEval=isEval, writeFile=False, verbose=verbose, clf=clf, clf_iter=clf_iter)
        if ins is None:
            logging.info("Skipping file %s - could not generate mol block!"%(f))
            continue
        mol = Chem.MolFromMolBlock(ins,sanitize=False)
        w.write(mol)
    logging.info("Writing multiple .xyz files to: %s"%(res_filename))
    w.close()

