#!/usr/bin/python
#
# Copyright 2018, Christoph Loschen
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other
# materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

from __future__ import print_function

from time import time
import os,sys,re,subprocess
import StringIO
import random

import logging
import argparse
import cPickle as pickle

from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Draw

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist,squareform
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold,GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sn

__author__ = 'chris'


def extract_features(mol, sourcename, pos, printHeader=True, fillNa=np.nan, xyz_file=None, plot=False, useSelectionRules=True, OrderAtoms=True, bondAngles=True, skipH=False, addBonds=False, verbose=False):
    """
    Create feature matrix from RDKit mol object or xyz file

    :param mol: RDKit molecule
    :param sourcename: name of sd file
    :param pos: position in sdf
    :param xyz_file: name of xyz
    :param plot: plotting
    :param useSelectionRules: use rules to remove strange bonds
    :param OrderAtoms: larger atomic number first
    :param bondAngles: add bond angles, i.e. distance to third atom
    :param skipH: remove H
    :param addBonds:  add neighbor bonds as features
    :param printHeader: prints column headers
    :param fillNa: how to fill NA values
    :param verbose: verbosity on/off

    :return: pandas dataframe with feature matrix
    """

    pt = Chem.GetPeriodicTable()

    if xyz_file is not None:
        if xyz_file.lower().endswith(".xyz"):
            atomtypes, coords, q, title = read_xyz(xyz_file, skipH=skipH)
        elif xyz_file.lower().endswith(".pdb"):
            atomtypes, coords, q, title = read_pdbfile(xyz_file, skipH=skipH)
        if q!=0:
            logging.info("Found charge: %.2f"%(q))
        dm = squareform(pdist(np.asarray(coords)))
    else:
        if skipH:
            try:
                mol = Chem.RemoveHs(mol)
            except ValueError as e:
                logging.info("Skipping H deletion for molecule at pos:" + str(pos))
                return(None)
        #check if bonds are available
        try:
            if not addBonds and mol.GetNumBonds(onlyHeavy=False)==0:
                logging.info("No bonds found: skipping molecule %s " %Chem.MolToSmiles(mol))
                return (None)
        except RuntimeError as e:
            logging.info("RuntimeError: skipping molecule")
            return(None)

        dm = Chem.Get3DDistanceMatrix(mol)  # both should be the same!!!
        q = Chem.GetFormalCharge(mol)

    n,m = dm.shape
    assert(n == m)

    if plot:
        plt.pcolormesh(dm)
        plt.colorbar()
        plt.xlim([0, n])
        plt.ylim([0, n])
        plt.show()

    dist_cut = 3.0  # distance cutoff
    n_cut = 3  # neighbour cutoff

    if printHeader and verbose:
        print('{:<4s}{:<4s}{:>4s}{:>3s}{:>3s}{:>8s}'.format('ID1','ID2','Q', '#1', '#2', 'DIST'),end='')
        for i in xrange(2*n_cut):
            if addBonds:
                print('{:>4s}{:>3s}{:>8s}{:>8s}{:>4s}'.format('POS', '#', 'DIST', 'DISTB','BNB'),end='')
            elif bondAngles:
                print('{:>4s}{:>3s}{:>8s}{:>8s}'.format('POS', '#', 'DIST','DISTB'),end='')
            else:
                print('{:4s}{:3s}{:8s}'.format('POS', '#', 'DIST'),end='')
        print("{:4s}".format('TYPE'))

    df = []
    index = []

    for i in xrange(0,n):
        if xyz_file is not None:
            bnd_at1 = atomtypes[i]
            bond_num1 = pt.GetAtomicNumber(bnd_at1)
        else:
            bnd_at1 = mol.GetAtomWithIdx(i)
            bond_num1 = bnd_at1.GetAtomicNum()
            bnd_at1 = bnd_at1.GetSymbol()

        for j in xrange(0,m):
            row = []
            if i >= j: continue
            bnd_dist = dm[i,j]
            if bnd_dist>dist_cut: continue

            bnd_type = 0
            if xyz_file is None:
                bnd_at2 = mol.GetAtomWithIdx(j)
                bond_num2 = bnd_at2.GetAtomicNum()
                bnd = mol.GetBondBetweenAtoms(i, j)
                if bnd is not None:
                    bnd_type = int(bnd.GetBondTypeAsDouble())
                    if bnd.GetIsAromatic():
                        bnd_type = 4
                else:
                    bnd_type = 0
                bnd_at2=bnd_at2.GetSymbol()
            else:
                bnd_at2 = atomtypes[j]
                bond_num2 = pt.GetAtomicNumber(bnd_at2)

            #sanity checks
            if xyz_file is None:
                # we accept very short bonds but give warning
                selstr = "Skipping"
                if not useSelectionRules:
                    selstr = "Keeping"
                if bnd_dist<0.75 and bnd_type>0:
                    logging.warn("Unreasonable short X-X bond (r<0.75): %r(%d) %r(%d) %4.2f type: %d from source: %s at pos: %d"%(bnd_at1,i+1,bnd_at2,j+1,bnd_dist,bnd_type,sourcename,pos))
                elif bnd_dist<1.1 and bond_num1>=6 and bond_num2>=6 and bnd_type>0:
                    logging.warn("Unreasonable short X-X bond (r<1.1): %r(%d) %r(%d) %4.2f type: %d from source: %s at pos: %d"%(bnd_at1,i+1,bnd_at2,j+1,bnd_dist,bnd_type,sourcename,pos))
                # in case of problems we discard whole molecule
                elif bnd_dist < 0.75 and (bond_num1 == 1 or bond_num2 == 1) and bnd_type == 0:
                    logging.warn("%s unreasonable short X-H distance w/o bond: %r(%d) %r(%d) %4.2f type: %d from source: %s at pos: %d" % (selstr,
                        bnd_at1,i+1, bnd_at2,j+1, bnd_dist, bnd_type,sourcename,pos))
                    if useSelectionRules: return (None)
                elif bnd_dist < 1.5 and bond_num1==6 and bond_num2==6 and bnd_type==0:
                    logging.warn("%s unreasonable short C-C distance w/o bond: %r(%d) %r(%d) %4.2f type: %d from source: %s at pos: %d" % (selstr,
                    bnd_at1,i+1, bnd_at2,j+1, bnd_dist, bnd_type,sourcename,pos))
                    if useSelectionRules: return(None)
                elif bnd_dist < 1.0 and bond_num1>=6 and bond_num2>=6 and bnd_type==0:
                    logging.warn("%s unreasonable short distance w/o bond: %r(%d) %r(%d) %4.2f type: %d from source: %s at pos: %d" % (selstr,
                    bnd_at1,i+1, bnd_at2,j+1, bnd_dist, bnd_type,sourcename,pos))
                    if useSelectionRules: return(None)
                # rather generous cutoff
                elif bnd_dist>1.8 and bond_num1==6 and bond_num2==6 and bnd_type>0:
                    logging.warn("%s unreasonable long C-C bond: %r(%d) %r(%d) %4.2f type: %d from source: %s at pos: %d"%(selstr,bnd_at1,i+1,bnd_at2,j+1,bnd_dist,bnd_type,sourcename,pos))
                    if useSelectionRules: return(None)

            #unique order
            if OrderAtoms and bond_num1<bond_num2:
                row.extend([j + 1, i + 1, q,bond_num2, bond_num1, bnd_dist])
                i_tmp,j_tmp = j,i
            else:
                row.extend([i + 1, j + 1, q,bond_num1, bond_num2, bnd_dist])
                i_tmp, j_tmp = i, j

            if verbose: print('{:<4d}{:<4d}{:4.1f}{:3d}{:3d}{:8.3f}'.format(i_tmp+1,j_tmp+1,q,bond_num1,bond_num2,bnd_dist),end='')

            # now iterate over neighbors of a and b and i.e. sort row a and b and concat, then skip i and j
            for a in [i_tmp,j_tmp]:
                row_sorted_a = np.argsort(dm[a,:])
                count = 0
                k = 0
                if len(row_sorted_a) > 2:
                    for nextn in row_sorted_a:
                        if nextn == j_tmp or nextn == i_tmp:
                            continue
                        if k==n_cut:break

                        dist = dm[a,nextn]
                        if xyz_file is None:
                            at = mol.GetAtomWithIdx(nextn)
                            num = at.GetAtomicNum()
                            at = at.GetSymbol()
                        else:
                            at = atomtypes[nextn]
                            num = pt.GetAtomicNumber(at)

                        if bondAngles:
                            other = i_tmp if a==j_tmp else j_tmp
                            distb = dm[other,nextn]

                            if addBonds:
                                bndb = mol.GetBondBetweenAtoms(a, nextn)
                                if bndb is not None:
                                    bnd_typeb = int(bndb.GetBondTypeAsDouble())
                                    if bndb.GetIsAromatic():
                                        #bnd_type=randint(1,2)
                                        bnd_typeb = 4
                                else:
                                    bnd_typeb = 0
                                row.extend([num, dist, distb,bnd_typeb])
                                if verbose:
                                    print('{:4d}{:>3d}{:8.3f}{:8.3f}{:4d}'.format(nextn+1,num,dist,distb,bnd_typeb),end='')
                            else:
                                row.extend([num, dist,distb])
                                if verbose:
                                    print('{:4d}{:>3s}{:3d}{:8.3f}{:8.3f}'.format(nextn+1,at,num,dist,distb),end='')
                        else:
                            row.extend([num, dist])
                            if verbose:
                                print('{:4d}{:>3s}{:3d}{:8.3f}'.format(nextn+1,at,num,dist),end='')

                        k += 1
                        count += 1

                # padding
                while count<n_cut:
                    count += 1
                    if verbose:
                        print('{:>4d}{:>3s}{:3d}{:8.3f}'.format(0,"NA", 0, fillNa),end='')
                    row.extend([0, fillNa])
                    if bondAngles:
                        row.extend([fillNa])

            if verbose:  print('{:4d}'.format( bnd_type),end='')
            row.append(bnd_type)
            df.append(row)

            index.append(sourcename + '_pos' + str(pos+1) + '_' + str(i_tmp + 1) + 'x' + str(j_tmp + 1))

    try:
        df = pd.DataFrame(df)
        colnames = ['id1','id2','q','ata','atb','distab','ata1','dista1','ata2','dista2','ata3','dista3','atb1','distb1','atb2','distb2','atb3','distb3','bond']
        if addBonds:
            colnames = ['id1', 'id2', 'q', 'ata', 'atb', 'distab', 'ata1', 'dista1', 'dista1b','bonda1', 'ata2', 'dista2',
                        'dista2b','bonda2', 'ata3', 'dista3', 'dista3b','bonda3',
                        'atb1', 'distb1', 'distb1a','bondb1', 'atb2', 'distb2', 'distb2a','bondb2', 'atb3', 'distb3', 'distb3a','bondb3', 'bond']
        elif bondAngles:
            colnames = ['id1', 'id2', 'q', 'ata', 'atb', 'distab', 'ata1', 'dista1','dista1b', 'ata2', 'dista2','dista2b', 'ata3', 'dista3','dista3b',
                        'atb1', 'distb1','distb1a', 'atb2', 'distb2','distb2a', 'atb3', 'distb3','distb3a','bond']

        if len(colnames)!=len(df.columns):
            logging.error("Mismatch in dataframe colums for %s - SMILES: %s"%(sourcename+'_pos'+str(pos+1), Chem.MolToSmiles(mol)))


        df.columns = colnames
        df.index = index

    except ValueError:
        #i.e. for empty dataframes
        df = None
    return df


def convert_sdf2dataframe(infile, outfile="moldat.csv", fillNa=np.nan, sanitize=True, tempsave=False, useSelectionRules=True, skipH=False, addBonds=True, sample=None, debug=False, verbose=False):
    """
    Generate training dataset from list of sd files
    sd file -> Pandas DataFrame

    :param infile: sd file used for training
    :param outfile: feature matrix as .csv file
    :param fillNa: fill value for NA positions
    :param sanitize: switch this off for special molecules RDKit cannot digest, should be True in order to have aromatic bonds
    :param tempsave: save temporary data
    :param useSelectionRules: apply rules to filter nonsense structures
    :param skipH: remove hydrogens
    :param addBonds: inject neighbor bonds to feature matrix
    :param sample: subsample dataset fraction [0-1]
    :param verbose: verbosity on/off

    :return: feature matrix as pandas dataframe
    """
    logging.info("Generating feature using RDKit matrix from: %s -- with options skipH (%r) iterative(%r) filterRubbish(%r) "%(infile,skipH,addBonds,useSelectionRules))
    if sample is not None:
        logging.info("Subsampling fraction %4.2f of dataset"%(sample))
        np.random.seed(42)
    df_new = None
    suppl = Chem.SDMolSupplier(infile,removeHs=skipH,sanitize=False)
    count=0
    for i,mol in enumerate(suppl):
        if sanitize:
            try:
                Chem.SanitizeMol(mol) #adding aromatic bonds...we may have a problem here
            except ValueError as e:
                logging.info("Skipping sanitization for molecule at pos:" + str(i+1))
                if debug:
                    w = Chem.SDWriter('tmp_pos'+str(i+1)+'.sdf')
                    w.write(mol)
                    w.close()

                # we cannot use it then...
        if mol is not None:
            if sample is not None and np.random.random_sample()>sample:
                continue
            if i>0:
                df_new = pd.concat([df_new, extract_features(mol, infile, i, verbose=verbose, printHeader=True, fillNa=fillNa, useSelectionRules=useSelectionRules, skipH=skipH, addBonds=addBonds)], axis=0)
            else:
                df_new = extract_features(mol, infile, i, verbose=verbose, printHeader=True, fillNa=fillNa, useSelectionRules=useSelectionRules, skipH=skipH, addBonds=addBonds)
            count += 1
        else:
            logging.info("SKIPPING molecule at pos:"+str(i+1))
            logging.error("SKIPPING molecule at pos:" + str(i+1))

    logging.info("Processed total of >%d< molecules" % (count))
    if df_new is not None and tempsave:
        logging.info("%3d Generated temp file: %s" % (i + 1, outfile))
        df_new.to_csv(outfile,index=True)
    if df_new is None:
        logging.info("ERROR: There was a problem generating the data!")

    logging.info("Bond types: \n%r"%(df_new['bond'].value_counts()))
    logging.info("Total bonds: %r\n" % (df_new['bond'].value_counts().sum()))
    return(df_new)


def convert_sdfiles2csv(file_list = [], base_dir='', outdat='train_dat.csv', method='UFF', skipH=False, addBonds=False, sample=0.25, verbose=False):
    """
    Allows for training use a list of filenames, for internal testing

    :param file_list: list of .sd files
    :param base_dir: location of those files
    :param outdat: .csv file with feature matrix and target vectors
    """

    finalf = outdat
    for i,f in enumerate(file_list):
        infile = base_dir+f
        if not os.path.isfile(infile):
            logging.critical("File not found:"+infile)
            logging.critical("CWD:"+os.getcwd())
            sys.exit(1)
        outfile = 'moldat_tmp.csv'
        if infile.endswith('.smi'):
            infile = convert_smiles2sdfile(smifile=infile, outdat=outfile, method=method, verbose=verbose)
            infile = infile.replace(".smi",".sdf")
        print(infile)
        df = convert_sdf2dataframe(infile=infile, outfile=outfile, fillNa=9999.0, skipH=skipH, addBonds=addBonds, sample=sample, verbose=verbose)
        if df is None: continue

        outstr = 'writing'
        mode = 'w'
        header = True

        if os.path.isfile(finalf):
            mode = 'a'
            header = False
            outstr = 'appending'
        with open(finalf, mode) as f:
            df.to_csv(f, header=header, index=True)
        print(df.head())
        logging.info("File: %3d - %s .csv file to: %s" % (i + 1, outstr, finalf))


def train_from_csv(filename, grid_search=False, useRF=False, plotClassifier=False, save_clf='clf.p',verbose=False):
    """
    Train bond data with sklearn classifier, final model gets pickled.

    :param filename: .csv file with feature matrix
    :param grid_search: Do a parameter search on grid

    :return: trained scikit-learn model
    """
    logging.info("Training data on dataset:")
    df = pd.read_csv(filename,index_col=0)
    if 'id1' in df.columns and 'id2' in df.columns:
        df.drop(['id1', 'id2'], axis=1,inplace=True)
    logging.info("Shape   : %d X %d"%(df.shape[0],df.shape[1]))
    logging.info("Features: %s" % (df.columns))
    # remove similar data
    logging.info("Droping duplicates...")
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
        model = RandomForestClassifier(n_estimators=250, max_depth=None, min_samples_leaf=5, n_jobs=-1,
                                       max_features=X.shape[1] / 2, oob_score=False)
    else:
        #model = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.01, max_depth=5, NA=0, subsample=.5,colsample_bytree=1.0, min_child_weight=5, n_jobs=4, objective='multi:softprob',num_class=5, booster='gbtree', silent=1, eval_size=0.0)
        #parameters = {'n_estimators': [2000], 'learning_rate': [0.01, 0.1, 0.001], 'max_depth': [5, 7],'subsample': [0.5]}
        model = GradientBoostingClassifier(n_estimators=1000,learning_rate=0.1,max_depth=5,verbose=1)
        parameters = {}


    if grid_search:
        #model.set_params(n_jobs=1)
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
    pickle.dump(model,open( save_clf, "wb" ))
    logging.info("Saving classifier as: %s"%(save_clf))
    return(model)


def train_job(filename, reset=True, eval=False, fmethod='UFF', skipH=False, iterative=False, sample=False, useRF=False,verbose=False):
    """
    Use either .sdf or .smi file to
    train from a new dataset or append data

    :param filename: name of .smi of .sd file
    :param reset: removes old training data

    """

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
        logging.info("Using Random Forest for training!")


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


def eval_job(filename, skipH=False, iterative=False,verbose=False):
    """
    Evaluation per! molecule

    :param filename: filename for evaluation
    :param skipH: omit hydrogen
    :param iterative: use 2nd classifier
    :param verbose: verbose mode
    :return: -
    """
    # iterate over mols of SDF
    # mol -> df -> bonds_predicted / bonds_true
    # make SDF -> extract features -> df -> bonds_predicted2
    # compare bonds_true & bonds_predicted2
    # generatePredictions with mol
    print("Evaluation run with option: noH(%r)" % (skipH))
    print("Loading classifier...")
    clf = pickle.load(open('clf.p', "rb"))
    if iterative:
        clf_iter = pickle.load(open('clf_iter.p', "rb"))
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


def evaluate(filename_test,filename_train='train_dat.csv',plotting=True,iterative=False,verbose=False):
    """
    Evaluate on dataset with known bond info, molecule accuracy is computed afterwards

    :param filename_test: name of .csv file with feature matrix and targets
    """
    df = pd.read_csv(filename_test,index_col=0)
    filename_train=None
    # shown train_data
    if filename_train is not None:
        logging.info("Analyze train data...")
        df_train = pd.read_csv(filename_train,index_col=0)
        print(df_train.shape)
        df_train['bondtype']=df_train['bond'].astype('category')

        df_train = df_train[df_train.ata==6]
        df_train = df_train[df_train.atb==6]
        if plotting:
            ax = sn.boxplot(x="bond", y="distab", data=df_train[['distab','bond']])
            ax.set(ylabel='C-C distance', xlabel='bond type')
            #ax.set(xticklabels=[])
            plt.show()

    logging.info("Evaluate data set: " + filename_test)
    logging.info("Loading classifier...")
    clf = pickle.load(open("clf.p", "rb"))

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
            # ok no coordinates/no dm how to get feature matrix...????

        if np.array_equal(df_sub['bond_pred'].values, df_sub['bond'].values):
            ok += 1
        else:
            # print("FALSE: %s"%(name))
            not_ok += 1
            mask = df_sub['bond_pred'] != df_sub['bond']
            idx = np.argmax(mask)
            false_indices.append(idx)

    acc = ok/float(all)
    print(false_indices)
    print("\nTOTAL: %5d OK: %5d WRONG: %5d Accuray: %6.3f"%(all,ok,not_ok,acc))

    return(X)


def evaluate_OB(filename='fullerene_ml.sdf', verbose=False):
    """
    Evaluation via Open Babel

    :param filename: sd file
    :param removeHs: use H or not (obabel reorders X-H bonds...)
    :param verbose:  True for verbose
    :return: -
    """
    logging.info("Evaluating %s via OBabel"%(filename))
    #if sanitize:
    #    print("WARNING: Switched ON sanitization!")
    #else:
    #    print("WARNING: Switched OFF sanitization!")

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
        #generate xyz for OB prediction without H
        myfile = StringIO.StringIO(xyz_str)
        #if removeHs:
        #cmd_call = ["obabel", "-d","-ixyz", "-osdf"]
        #else:
        cmd_call = ["obabel", "-ixyz", "-osdf"]
        p = subprocess.Popen(cmd_call, stdin=subprocess.PIPE, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        molblock, err = p.communicate(myfile.read())
        #switch off sanitization
        #mol_pred_H = Chem.MolFromMolBlock(molblock,removeHs=False,sanitize=False)
        #always switch off H for comparison of main element bonds only
        mol_pred = Chem.MolFromMolBlock(molblock,removeHs=True,sanitize=False)

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
                res = raw_input()
                if 'n' in res.lower() or "f" in res.lower():
                    nfalse += 1
                    print("FALSE: %d/%d" % (nfalse,len(suppl)))
                # img.save('images/' + cname + '_' + str(i) + '.png')
                else:
                    nok += 1
                    print("OK: %d/%d" % (nok,len(suppl)))
                if verbose:
                    with open('ob_failure'+str(i+1)+'.sdf', 'w') as f:
                        f.write(molblock)
                    with open('ob_reference'+str(i+1)+'.sdf', 'w') as f:
                        f.write(Chem.MolToMolBlock(mol))
            else:
                #check for ambigious double bonds of C-O,N-O and P-O
                idx = np.where(bond_orig.values != df['bond'].values)
                for k in idx[0]:
                    try:
                        ata = df_orig.iloc[k][['ata']].values[0]
                        atb = df_orig.iloc[k][['atb']].values[0]
                    except IndexError as e:
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
                        f.write(molblock)
                    with open('ob_reference'+str(i+1)+'.sdf', 'w') as f:
                        f.write(Chem.MolToMolBlock(mol))
                    print("FALSE: %d/%d (%6.3f%%)" % (nfalse, len(suppl), 1.0-nfalse/(float) (i+1)))

    acc = nok / float(nall)
    logging.info("\nTOTAL: %5d OK: %5d WRONG: %5d Accuray: %6.3f" % (nall, nok, nfalse, acc))


def generate_multi_predictions(filename_list, skipH=False, iterative=False, forceAromatics=False, maxiter=1, isEval=False, verbose=False, **kwargs):
    """
    Generate predictions, i.e. sd file from list of xyz files

    :param filename_list: list of filenames
    :param skipH:  omit hydrogen
    :param iterative: 2-step prediction
    :param forceAromatics: deprecated
    :param maxiter: maxiteraions
    :param isEval: evaluation run with known bonds
    :param verbose: verbose
    :param kwargs: additional args
    :return: -
    """

    clf = pickle.load(open('clf.p', "rb"))
    if iterative:
        clf_iter = pickle.load(open('clf_iter.p', "rb"))
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


def generate_predictions(filename, skipH=False, iterative=False, forceAromatics=False, maxiter=1, isEval=False, writeFile=False, verbose=True, **kwargs):
    """
    Generate predictions for single molecule

    :param filename: single filename
    :param skipH:  omit hydrogen
    :param iterative: 2-step prediction
    :param forceAromatics: deprecated
    :param maxiter: maxiteraions
    :param isEval: evaluation run with known bonds
    :param verbose: verbose
    :param kwargs: additional args
    :return: sd molblock as string
    """

    clf = None
    clf_iter = None
    if kwargs is not None and 'clf' in kwargs.keys():
        #print(kwargs.keys())
        clf = kwargs['clf']
        clf_iter = kwargs['clf_iter']

    if iterative and not os.path.isfile("clf_iter.p"):
        sys.stderr.write("ERROR: Please train classifier in iterative mode (--iterative) first!\n")
        sys.exit(1)
    elif iterative:
        if clf_iter is None:
            print("Loading classifier clf_iter.p...")
            clf_iter = pickle.load(open('clf_iter.p', "rb"))

    #from evaluation
    if isinstance(filename, Chem.rdchem.Mol):
        mol = filename
        filename = 'eval'
        atomtypes, coords, q, title = read_mol(mol)
        X = predict_bonds(mol, q=q, skipH=skipH, iterative=False, forceAromatics=forceAromatics, clf=clf, eval=True, verbose=False)
        if X is None: return(None)

    elif not filename.lower().endswith('.xyz') and not filename.lower().endswith('.pdb'):
        sys.stderr.write("ERROR: Need .xyz/.pdb file for prediction!\n")
        sys.exit(1)

    elif not os.path.isfile("clf.p"):
        sys.stderr.write("ERROR: Please train classifier first!\n")
        sys.exit(1)

    elif filename.lower().endswith(".xyz"):
        atomtypes, coords, q, title = read_xyz(filename, skipH=skipH)
        X = predict_bonds(filename, q=q, skipH=skipH, iterative=False, forceAromatics=forceAromatics, clf=clf, verbose=verbose)

    elif filename.lower().endswith(".pdb"):
        atomtypes, coords, q, title = read_pdbfile(filename, skipH=skipH)
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
            #cluster similar bonds and flip them together...!!!
            if verbose: print(X2.head(40))
            #create probability gradient
            #dX2['dp(X)'] = X2['p(X)'] - X['p(X)'].values

            dX2 = pd.DataFrame(index=X2.index)
            dX2['dp(-)'] = X2['p(-)'] - X['p(-)'].values
            dX2['dp(=)'] = X2['p(=)'] - X['p(=)'].values
            dX2['dp(#)'] = X2['p(#)'] - X['p(#)'].values
            dX2['dp(a)'] = X2['p(a)'] - X['p(a)'].values
            grad = dX2.values
            #dX2['rand'] = np.random.beta(5.0,1.0,dX2.shape[0])
            #dX2['rand'] = np.random.ranf(dX2.shape[0])
            dX2['update'] = np.max(grad, axis=1)>0.25

            #dX2['keep'] = np.max(grad, axis=1) < 0.5
            mask = dX2['update'].values
            #what to do here???? switch only update one after another
            #charge??
            idx = np.where(mask==True)[0]
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
                return (True)
            else:
                return (False)

    if writeFile:
        # sdf format does not know delocalized charge...!?
        filename = os.path.basename(filename).replace('.xyz', '_ml.sdf').replace('.pdb', '_ml.sdf')
        with open(filename, 'w') as f:
            f.write(ins)

        logging.info("ML-generated SDF written to: " + filename)
    elif isEval:
        #
        if np.array_equal(X.bond_orig.values, X['bond'].values):
            return (True,ins)
        else:
            if verbose: print(X[['bond', 'bond_orig']])
            return (False,ins)
    else:
        return(ins)


def predict_bonds(filename, q=0, skipH=False, iterative=False, forceAromatics=False, clf=None, eval=False, verbose=False):
    """
    Create dataframe with bond info for 1 molecule
    Either from SDF/mol or from xyz_file
    :return: pandas dataframe with bond info
    """

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
                logging.info("Could not generate dataframe from %s"%(filename))
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
    if verbose: print("X shape n=%d m=%d "%(X.shape[0],X.shape[1]))


    if clf is None:
        print("Loading classifier %s..." % (clf_name))
        clf = pickle.load(open(clf_name, "rb"))
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
        sys.stderr.write("I/O error{0}\n".format(e))
        sys.stderr.write("Are there have enough training data for all bond types?\n")
        sys.exit(1)

    X = X[['id1','id2','q', 'ata', 'atb', 'distab', 'bond', 'p(-)', 'p(=)', 'p(#)', 'p(a)','p(X)','S']]

    # check high aromaticity case
    if forceAromatics:
        logging.info("Forcing aromaticity for bonds with p(a)>0.4!")
        potaroma_idx = X['p(a)'] > 0.4
        X.loc[potaroma_idx, 'bond'] = 4

    if eval:
        X['bond_orig'] = bond_orig

    if skipH: X = X[(X.ata != 1) | (X.atb == 1)]

    return(X)


def convert_smiles2sdfile(smifile="./smi_repository/bbp2.smi", method='UFF', outdat='train_dat.csv'):
    """
    Generates 3D sd file using distance geometry and force-field optimisation


    :param smifile: .smi file used as basis for training
    :param method:  UFF/MMFF/conf
    :param outdat: filename of training data

    """
    logging.info("Selected optimization method: %s"%(method))
    mols = Chem.SmilesMolSupplier(smifile,delimiter='\t ',titleLine=False, sanitize=True)
    mols = [x for x in mols if x is not None]

    sdfile = smifile.replace('.smi', '_' + method + '.sdf')
    w = Chem.SDWriter(sdfile)

    for i,m in enumerate(mols):
        #remove multiple molecule
        tmp = Chem.GetMolFrags(m,asMols=True)
        if (len(tmp)>1):
            logging.info("Using only first fragment: %s"%(Chem.MolToSmiles(m)))
            m = tmp[0]
        # add H for SDF file
        m = Chem.AddHs(m)
        conv = 0
        try:
            if method=='UFF':
                Chem.EmbedMolecule(m)
                conv = Chem.UFFOptimizeMolecule(m,maxIters=200)
            elif method=='MMFF':
                Chem.EmbedMolecule(m)
                conv = Chem.MMFFOptimizeMolecule(m,maxIters=200)
            elif method=='ETKDG':
                Chem.EmbedMolecule(m, Chem.ETKDG())
            else:
                Chem.EmbedMultipleConfs(m, 10, Chem.ETKDG())

            #also add not converged molecules!!!
            if conv==0 or conv==1:
                # add aromaticity flags again!
                Chem.SanitizeMol(m)
                smiles = Chem.MolToSmiles(Chem.RemoveHs(m))
                m.SetProp("SMILES", smiles)
                w.write(m)
            if conv==1:
                logging.info("Optimization not converged for molecule: %d with SMILES: %s" % (i, Chem.MolToSmiles(Chem.RemoveHs(m))))
            elif conv==-1:
                logging.info("Forcefield could not be setup for molecule: %d with SMILES: %s" % (i, Chem.MolToSmiles(Chem.RemoveHs(m))))

        except ValueError:
            logging.info("Optimization problem for molecule: %d with SMILES: %s"%(i,Chem.MolToSmiles(Chem.RemoveHs(m))))

    w.close()
    logging.info(("Writing to SD file:"+sdfile))
    return sdfile


def shuffle_split_sdfile(infile='test.sdf', frac=0.8, rseed=42):
    """
    Shuffle and split molecules within SDF for train / test set generation

    :param infile:
    :return:
    """
    random.seed=rseed
    trainfile = infile.replace('.sdf', '_train.sdf')
    w1 = Chem.SDWriter(trainfile)
    w1c=0
    testfile = infile.replace('.sdf', '_test.sdf')
    w2 = Chem.SDWriter(testfile)
    w2c=0
    suppl = Chem.SDMolSupplier(infile, removeHs=False, sanitize=False)
    for i, mol in enumerate(suppl):
        rdm = random.random()
        if rdm<frac:
            w1.write(mol)
            w1c+=1
        else:
            w2.write(mol)
            w2c+=1

    w1.close()
    w2.close()
    print("Finished writing %d train moles / %d test moles"%(w1c,w2c))


def get_stats(infile,skipH=False,addBonds=False):
    """
    Statistical analysis of SD file

    :param filename: name of .sdf
    """

    df = convert_sdf2dataframe(infile=infile, outfile=None, fillNa=9999.0, verbose=False, skipH=skipH,
                               addBonds=addBonds)
    df = df[df.bond>0]
    bonds=[['C','C'],['O','C'],['N','C'],['P','C'],['S','C'],['P','O'],['S','O']]
    n_rows = 2
    n_cols = 4

    pt = Chem.GetPeriodicTable()
    fig = plt.figure()
    plt.suptitle(infile)
    for i,(a,b) in enumerate(bonds):
        title = str(a) + '&' + str(b)
        a = pt.GetAtomicNumber(a)
        b = pt.GetAtomicNumber(b)

        idx = ((df.ata.values == a) & (df.atb.values == b)) | ((df.ata.values == b) & (df.atb.values == a))
        df_tmp = df[idx]

        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        ax.set_title(title)
        ax.set_xlabel('dist')
        #colors = ['blue','black','cyan','green']
        data = [df_tmp.distab[df_tmp.bond==1],df_tmp.distab[df_tmp.bond==2],df_tmp.distab[df_tmp.bond==3],df_tmp.distab[df_tmp.bond==4]]
        labels = ['-','=','#','a']
        #colors = ['blue']
        #data = [df_tmp.distab[df_tmp.bond == 1]]
        #labels = ['-']

        ax.hist(data,bins=20, alpha=0.5,stacked=True,histtype='bar',label=labels)
        fig.legend(labels=labels,loc = 'upper right', ncol=5, labelspacing=0.)
        #df_tmp.groupby('bond').distab.hist(bins=40, ax=ax,sharex=True,sharey=True, alpha=[0.2,0.2,0.2,0.2])
        #df_tmp.distab.hist(bins=40,by=df_tmp.distab, ax=ax, sharex=True, sharey=True, alpha=[0.2, 0.2, 0.2, 0.2])
        #df_tmp.groupby('bond').distab.plot(kind='kde', ax=ax)
    fig.tight_layout()
    plt.show()


def clean_sdf(infile='test.sdf'):
    """
    Cleans SD file from unreasonable structures via rules from extractFeatures function

    :param infile: .sdf to clean
    """
    outfile = infile.replace('.sdf', '_clean.sdf')
    w = Chem.SDWriter(outfile)
    suppl = Chem.SDMolSupplier(infile, removeHs=False, sanitize=False)
    count=0
    for i, mol in enumerate(suppl):
        df = extract_features(mol, infile, i, verbose=False, printHeader=True,
                              useSelectionRules=True)
        if df is not None and mol is not None:
            w.write(mol)
        else:
            count+=1

    logging.info("Removed %d files - saving .sdf to: %s "%(count,outfile))
    w.close()


def sanitize_sdf(infile='test.sdf',removeHs=False):
    """
    Sanitize SD file

    :param infile: .sdf to clean

    """
    outfile = infile.replace('.sdf', '_sane.sdf')
    w = Chem.SDWriter(outfile)
    suppl = Chem.SDMolSupplier(infile, removeHs=removeHs, sanitize=True)
    count=0
    for i, mol in enumerate(suppl):
        if mol is not None:
            w.write(mol)
        else:
            count+=1

    logging.info("Removed %d files - saving .sdf to: %s "%(count,outfile))
    w.close()


def read_pdbfile(filename, skipH=False):
    """
    Read pdb data

    :param filename: filename of .pdb file
    :param skipH: Do not read H atoms
    :return: atomtypes, coordinates and title section
    """

    mol = Chem.MolFromPDBFile(filename)
    atomtypes, coords, q, title = read_mol(mol)
    return(atomtypes, coords, q, title)


def read_mol(mol):
    """
    Analyze RDKit mol

    :param mol:
    :return: atomtypes, coordinates, charge & title
    """
    title = ""
    coords = []
    atomtypes = []
    for i, atom in enumerate(mol.GetAtoms()):
        an = atom.GetSymbol()
        if an == 1:
            logging.warn("PDB file should not contain hydrogens!")
        atomtypes.append(an)
        pos = mol.GetConformer().GetAtomPosition(i)
        coords.append([pos.x, pos.y, pos.z])

    q = Chem.GetFormalCharge(mol)
    return (atomtypes, coords, q, title)

def read_xyz(filename, skipH=False):
    """
    Read xyz data

    :param filename: filename of .xyz file
    :param skipH: Do not read H atoms
    :return: atomtypes, coordinates and title section

    https://github.com/pele-python/pele/blob/master/pele/utils/xyz.py
    """
    with open(filename,'r') as fin:
        natoms = int(fin.readline())
        title = fin.readline()[:-1]
        q=0
        qin = re.search("(?:CHARGE|CHG)=([-+]?\d*\.\d+|\d+|[-+]?\d)",title,re.IGNORECASE)
        if qin:
            q = float(qin.group(1))
        coords = []
        atomtypes = []
        for x in xrange(natoms):
            line = fin.readline().split()
            if (line[0].lower()=='h') and skipH: continue
            atomtypes.append(line[0])
            coords.append([float(line[1]),float(line[2]),float(line[3])])

    return(atomtypes,coords, q, title)

def create_sdfile(name, atomtypes, coords, df):
    """
    Creates string with SD info

    :param name: molecule name
    :param atomtypes: atomic types
    :param coords:  coordinates
    :param df:  dataframe with ids and bond types
    :return: molblock
    """
    df = df[df.bond>0]

    ins = name + "\n"

    # comment block
    ins += "ML generated sdf\n"
    ins += "\n"
    ins += "%3d%3d  0  0  0  0  0  0  0  0  1 V2000\n" % (len(atomtypes), (df.bond > 0).sum())

    # atomb block
    for at, xyz in zip(atomtypes, coords):
        ins += "%10.4f%10.4f%10.4f %-2s 0  0  0  0  0\n" % (xyz[0], xyz[1], xyz[2], at)
        # ins += "%2s %12.4f %12.4f %12.4f  \n" % ( at,xyz[0], xyz[1], xyz[2])

    # bond block
    for index, row in df.iterrows():
        ins += "%3d%3d%3d  0  0  0  0\n" % (row['id1'], row['id2'], row['bond'])

    ins += "M  END"
    return(ins)


def mol_dataframe_generator(X):
    # modfiy index to isolate molecule id
    X['index_mod'] = X.index.str.split("_pos")
    X['index_mod'] = X.index_mod.str[1]
    X['index_mod'] = X.index_mod.str.split("_")
    X['index_mod'] = X.index_mod.str[0]
    grouped = X.groupby('index_mod')
    for name, df_sub in grouped:
        yield name,df_sub


def mol2xyz(mol):
    """
    Converts RDKit mol to xyz
    :param mol:  RDKit mol
    :return: xyz string

    """
    atomtypes, coords, q, title = read_mol(mol)

    xyz_str = "%d\n\n"%(len(atomtypes))
    for at, xyz in zip(atomtypes, coords):
        xyz_str += "%3s%10.4f%10.4f%10.4f\n" % (at,xyz[0], xyz[1], xyz[2])

    return(xyz_str)

def plot_classification_results(y, ypred):
    """
    Plots classification results

    :param y: ground truth
    :param ypred: prediction

    """
    report = classification_report(y, ypred, digits=3, target_names=['X', '-', '=', '#', 'a'])
    print(type(report))
    print(report)

    cm = confusion_matrix(y, ypred, labels=[0, 1, 2, 3, 4])
    df_cm = pd.DataFrame(cm, index=[i for i in "X-=#a"],
                         columns=[i for i in "X-=#a"])
    plt.figure(figsize=(10, 7))
    sn.set(font_scale=1.4)  # for label size
    ax = sn.heatmap(df_cm, annot=True, fmt='d', annot_kws={"size": 16}, cmap="magma_r")  # font size

    ax.set(xlabel='predicted bond type', ylabel='true bond type')
    plt.show()

def set_logging(loglevel = logging.INFO):
    """
    Sets up logging

    :param loglevel
    """
    logging.basicConfig(filename='log.txt', level=loglevel,format='%(asctime)s - %(levelname)s - %(message)s')
    root = logging.getLogger()
    root.setLevel(loglevel)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(loglevel)
    formatter = logging.Formatter('%(message)s')
    ch.setFormatter(formatter)
    root.addHandler(ch)

if __name__ == "__main__":
    import rdkit

    t0 = time()
    pd.set_option('display.max_columns', 30)
    pd.set_option('display.width', 1000)

    np.random.seed(42)
    set_logging()

    description = """
    MAMBA - MAchine-learning Meets Bond Analysis
       
    Creates .sdf file incl. chemical bond information out of a .xyz file
    Bond perception is learned via machine learning (scikit-learn classifier) from
    previous .sdf or .smi files.
    
    Internally uses RDKit, numpy, pandas and scikit-learn python packages
    
    (c) 2018 Christoph Loschen
    
    Examples:
    
    1) mamba.py --train largefile.sdf
    2) mamba.py --add special_case.sdf [optional]
    3) mamba.py --predict new_molecule.xyz
    
    
    """
  
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    help_str= """
    OK
    """

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-t','--train',action='store_true', help="train   - train the parser using a .sdf or .smi file")
    group.add_argument('-a','--add',action='store_true', help='add     - add data to the parser using a .sdf or .smi file')
    group.add_argument('-p','--predict',action='store_true',help='predict - create .sdf file using .xyz file as input\n')
    group.add_argument('--eval',action='store_true', help='eval - evaluate .sdf file\n')
    group.add_argument('--evalOB', action='store_true', help='eval - evaluate .sdf file via openbabel \n')
    group.add_argument('--analyze',action='store_true', help='analyze - analyze last evulated structure\n')
    group.add_argument('--stats', action='store_true', help='statistics on SD file molecules\n')

    group.add_argument('--clean', action='store_true', help='clean .sd file from unreasonable structures\n')
    group.add_argument('--sanitize', action='store_true', help='sanitize (via RDKit) .sd file from unreasonable structures\n')
    group.add_argument('--shufflesplit', type=float, default=0.8, metavar='f',help='Create test and train set by splitting and shuffling molecules, f is a float [0..1] \n')

    parser.add_argument('--noH', action='store_true', help='omit Hydrogen atom when learning\n',default=False)
    parser.add_argument('--useRF', action='store_true', help='use Random Forest instead of Gradient Boosting for training\n', default=False)
    parser.add_argument('-v','--verbose', action='store_true', help='verbose\n', default=False)
    parser.add_argument('--iterative', action='store_true', help='Iterative prediction of bonds using 2 classifiers\n', default=False)
    parser.add_argument('--sample', type=float, default=None,metavar='f',help='Subsampling of dataset during training, f is a float [0..1] \n')
    parser.add_argument('--FF', choices=("UFF", "MMFF", "ETKDG"), help='Forcefield/Method to use for 3D structure generation from SMILES', required=False,default="UFF")

    parser.add_argument("filename", nargs='+', help=".sdf or .smi for training, .xyz for prediction",default=[sys.stdin])

    fmethod = 'UFF'

    args = parser.parse_args()
    if args.FF!='UFF':
        fmethod=args.FF

    if args.verbose:
        print("Verbose ON")

    if len(args.filename)>1:
        if args.predict and (args.filename[0].endswith(".xyz") or args.filename[0].endswith(".pdb")):
            generate_multi_predictions(args.filename, iterative=args.iterative)
        else:
            print("Multiple files only allowed for prediction with .xyz and .sdf")
            sys.exit(1)
    else:
        for f in args.filename:

            if args.train:
                train_job(f, reset=True, fmethod=fmethod, skipH=args.noH, iterative=args.iterative, useRF=args.useRF, sample=args.sample)

            elif args.add:
                train_job(f, reset=False, fmethod=fmethod, skipH=args.noH, iterative=args.iterative, useRF=args.useRF, sample=args.sample)

            elif args.predict:
                generate_predictions(f, iterative=args.iterative,writeFile=True, verbose=args.verbose)

            elif args.eval and args.iterative:
                eval_job(f, skipH=args.noH, iterative=args.iterative,verbose=args.verbose)

            elif args.eval:
                train_job(f, eval=True, fmethod=fmethod, skipH=args.noH, iterative=args.iterative, sample=args.sample,verbose=args.verbose)

            elif args.evalOB:
                evaluate_OB(f,verbose=args.verbose)

            elif args.analyze:
                evaluate('eval_dat.csv',plotting=True)

            elif args.stats:
                get_stats(f)

            elif args.clean:
                clean_sdf(f)

            elif args.sanitize:
                sanitize_sdf(f)

            elif args.shufflesplit:
                shuffle_split_sdfile(f, frac=args.shufflesplit)

    print("FINSIHED in %fs\n" % (time() - t0))



