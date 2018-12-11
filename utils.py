#!/usr/bin/python

import logging
import os,sys,re,subprocess
import cPickle as pickle

import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt
import seaborn as sn

from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Draw

from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.manifold import TSNE

from mamba import generate_predictions,convert_sdf2dataframe

__author__ = 'chris'

def cosine_law(a,b,c):
    """
    law of cosine

    :param a: distance1
    :param b: distance2
    :param c: distance3
    :return: angle in C-A-B
    """
    if not (a>1E-5 and b>1E-5):
        logging.warn("Found very small distance: a: %6.3f b: %6.3f"%(a,b))
    acosy = min(max(-1.0,(a**2+b**2-c**2)/(2*a*b)),1.0)
    acosy = math.acos(acosy)
    return(acosy)

def check_cosine_law_prediction():
    """
    Simple test fun
    d1 = 2.147371
    d2 = 2.121562
    d3 = 3.155758
    alpha = cosine_law(d1,d2,d3)
    alpha_deg = math.degrees(alpha)
    print("alpha(radians): %6.3f alpha(degree): %6.3f"%(alpha,alpha_deg))
    """
    print("Checking law of cosines...")
    df = pd.read_csv('cos.csv',index_col=0)
    df = df[['distab','dista1','dista1b','dista2','dista3']]
    print("Creating predictions...")
    y = df.apply(lambda x: math.degrees(cosine_law(x['distab'],x['dista1'],x['dista1b'])),axis=1)
    #df = df.drop(['dista1b'],axis=1)
    #y = df.apply(lambda x: cosine_law(x['distab'], x['dista1'], x['dista1b']), axis=1)
    print("Fitting model...")
    #model = RandomForestRegressor(n_estimators=250,n_jobs=1)
    #model = GradientBoostingRegressor(n_estimators=1000,learning_rate=0.1, max_depth=5)
    model = XGBRegressor(n_estimators=1000, learning_rate=0.1, max_depth=5, NA=0, subsample=1.0,
                              colsample_bytree=1.0, min_child_weight=5, n_jobs=4, objective='reg:linear',
                              booster='gbtree', silent=1, eval_size=0.0)

    model.fit(df,y)
    ypred = model.predict(df)
    rmse = mean_squared_error(ypred,y)**0.5
    print("RMSE: %6.3f"%(rmse))
    sn.set_style("white")
    x, y = pd.Series(ypred, name=r'learned angle $\alpha$'), pd.Series(y, name=r'exact angle $\alpha$')
    ax = sn.regplot(x=x, y=y, marker="o",color='black')
    plt.show()


    """
    smiles = 'CC([O-])=O'
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    Chem.EmbedMolecule(mol)
    Chem.UFFOptimizeMolecule(mol)
    extractFeatures(mol, smiles, 0, verbose=True, printHeader=True, fillNa=np.nan, xyz_file=None, plot=False,
                    useSelectionRules=True, OrderAtoms=True, bondAngles=True)
    """
    sys.exit(1)

def clean_sdf(infile='/home/loschen/calc/ml_bond_parser/pdbbind_refined/sdf/pdbbind_refined.sdf'):
    """
    Cleans SD file from unreasonable structures via rules from extractFeatures function

    :param infile: .sdf to clean
    """
    outfile = infile.replace('.sdf', '_clean.sdf')
    w = Chem.SDWriter(outfile)
    suppl = Chem.SDMolSupplier(infile, removeHs=False, sanitize=False)
    count=0
    not_sane=0
    ok=0
    total=0
    for i, mol in enumerate(suppl):
        total+=1
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            count += 1
            not_sane += 1
            continue
        if mol is not None:
            w.write(mol)
            ok+=1
        else:
            count+=1

    print("Inspected %d files. Removed %d files not sane: %d - saving %d files as .sdf to: %s "%(total,count,not_sane,ok,outfile))
    w.close()


def centerMol(mol):
    from rdkit.Chem import rdMolTransforms
    conf = mol.GetConformer()
    pt = rdMolTransforms.ComputeCentroid(conf)
    for i in range(conf.GetNumAtoms()):
        conf.SetAtomPosition(i,conf.GetAtomPosition(i) - pt)
    return(mol)


def pdb2sdf_via_templates2(noLabute=True,smidir='/home/loschen/calc/ml_bond_parser/naomi_rarey/ci300358c_si_001/smiles/', pdbdir='/home/loschen/calc/ml_bond_parser/naomi_rarey/ci300358c_si_001/pdb/',inverse=True):
    """

    Converts no H/no bond pdb to proper sdf using template smi
    Starting from pdb

    :param smifile:
    :param pdbdir:
    :return:

    """
    from os import listdir
    from os.path import join, isfile

    sdfile = './naomi_rarey/ci300358c_si_001/naomi_viaRDKIT_nolabute.sdf'
    w = Chem.SDWriter(sdfile)

    pdbfiles = [f for f in listdir(pdbdir) if isfile(join(pdbdir, f)) and f.endswith('.pdb')]

    f = open("/home/loschen/calc/ml_bond_parser/labute_set/labute.smi","r")
    labute_names=[]
    for x in f:
        smi = x.split()
        if len(smi)>1:
            labute_names.append(smi[1].strip())
    labute_short = [x[:8] for x in labute_names]

    print("Labute names: %d"%(len(labute_names)))
    raw_input()
    smi_missed = 0
    assignment_failed = 0
    converted =0
    skipped=0
    for f in pdbfiles:
        cname = f.replace('.pdb','')
        smifile = smidir + cname + '.smi'
        if not os.path.isfile(smifile):
            smi_missed+=1
        else:
            print smifile
            refmoles = Chem.SmilesMolSupplier(smifile,delimiter='\t ',titleLine=False, sanitize=True)
            mol = Chem.MolFromPDBFile(pdbdir+f)
            if mol is None:
                continue

            for ref in refmoles:
                if ref is None: continue
                try:
                    mol = Chem.AssignBondOrdersFromTemplate(ref, mol)
                    break
                except ValueError:
                    assignment_failed+=1
                    continue
            mol = centerMol(mol)
            mol.SetProp("_Name", cname)
            Chem.SanitizeMol(mol)
            if cname in labute_names and noLabute:
                print cname
                skipped +=1
            elif cname[:8] in labute_short and noLabute:
                print cname,labute_short[labute_short.index(cname[:8])]
                skipped += 1
            else:
                w.write(mol)
                converted += 1
    w.close()
    print("\nConversion finished: %d molecules, %d smi missed, %d assignments failed, %d converted %d skipped." % (
        len(pdbfiles), smi_missed, assignment_failed, converted,skipped))


def pdb2sdf_via_templates(smifile="labute_set/labute.smi", pdbdir="labute_set/pdb_rarey/"):
    """
    Converts no H/no bond pdb to proper sdf using template smi
    Starting frmo smiles

    :param smifile: smifile
    :param pdbdir:  directory with pdb needs identical names
    """

    refs = Chem.SmilesMolSupplier(smifile,delimiter='\t ',titleLine=False, sanitize=True)
    sdfile = smifile.replace('.smi', '_viaRDKIT.sdf')
    w = Chem.SDWriter(sdfile)

    pdb_missed=0
    assignment_failed=0
    converted=0
    for ref in refs:
        cname = ref.GetProp("_Name")
        pdbfile = pdbdir+cname+'.pdb'
        similar=True
        if not os.path.isfile(pdbfile):
            pdb_missed += 1
            files = os.listdir(pdbdir)
            similar=False
            for file in files:
                if cname[:5] in file:
                    logging.info("Similar PDB found: %s & %s"%(file,pdbfile))
                    pdb_missed -= 1
                    pdbfile=pdbdir+file
                    similar=True
        if similar:
            logging.warn("PDB OK: %s"%pdbfile)
        else:
            logging.warn("PDB file not found: %s" % pdbfile)
            continue

        mol = Chem.MolFromPDBFile(pdbfile)
        if mol is None:
            assignment_failed += 1
            continue
        try:
            mol = Chem.AssignBondOrdersFromTemplate(ref, mol)
        except ValueError:
            assignment_failed += 1
            continue
        mol = centerMol(mol)
        mol.SetProp("_Name",cname)
        Chem.SanitizeMol(mol)
        w.write(mol)
        converted+=1
    w.close()
    print("\nConversion finished: %d molecules, %d pdb missed, %d assignments failed, %d converted."%(len(refs),pdb_missed,assignment_failed,converted))
    sys.exit(1)

def evaluate_labute(smifile="labute_set/labute.smi",pdbdir='/home/loschen/calc/ml_bond_parser/naomi_rarey/ci300358c_si_001/pdb/',iterative=False,skipH=True,showImages=True):
    """
    Evaluate labute files

    :param smifile:
    :param pdbdir:
    :param iterative:
    :param skipH:
    :param showImages:
    :return:
    """
    showImages=False
    pdbdir = "labute_set/pdb_rarey/"
    refs = Chem.SmilesMolSupplier(smifile, delimiter='\t ', titleLine=False, sanitize=True)
    labute_set = []
    missing = 0
    hits = 0
    for ref in refs:
        similar = False
        same = False
        cname = ref.GetProp("_Name")

        #print("Name: %8s"%(cname))
        pdbfile = pdbdir + cname + '.pdb'
        if not os.path.isfile(pdbfile):
            files = os.listdir(pdbdir)
            similar=False
            for file in files:
                if cname[:8] in file and file.endswith('pdb'):
                    pdbfile=pdbdir+file
                    similar=True
        else:
            same=True

        if same or similar:
            print("%-10s: PDB found: %s" % (cname, pdbfile))
            labute_set.append([cname,ref, Chem.MolToSmiles(ref),pdbfile])
        else:
            print("%-10s: missing!!" % (cname))
            missing+=1

    print("Evaluation run with option: noH(%r)" % (skipH))
    print("Loading classifier...")
    clf = pickle.load(open('clf.p', "rb"))
    if iterative:
        clf_iter = pickle.load(open('clf_iter.p', "rb"))
    else:
        clf_iter = None

    res_dict={}
    assignment_failed =0
    for i,(cname,ref,smiles,pdbfile) in enumerate(labute_set):
        res_dict[cname]=0
        #create SD file via template
        mol = Chem.MolFromPDBFile(pdbfile,sanitize=False,removeHs=True)
        if mol is None or ref is None:
            assignment_failed += 1
            continue
        try:
            mol = Chem.AssignBondOrdersFromTemplate(ref, mol)
        except ValueError:
            assignment_failed += 1
            continue
        if mol is None:
            print("%d %s - Could not create mol from PDB!"%(i,cname))
            continue
        res,sdf_pred = generate_predictions(mol, skipH=skipH, iterative=iterative, forceAromatics=False, maxiter=1, verbose=False,
                                            clf=clf, clf_iter=clf_iter, isEval=True)
        #show 2D pictures of mol & res_sdf
        if showImages:
            mol_pred = Chem.MolFromMolBlock(sdf_pred,sanitize=True)
            if mol_pred is None:
                print("WARNING: Could not sanitize predicted mol!")
                continue
            Chem.Compute2DCoords(mol)
            mol_pred = Chem.AddHs(mol_pred)
            mol_pred = Chem.RemoveHs(mol_pred)
            Chem.Compute2DCoords(mol_pred)
            ms = [mol,mol_pred]
            res_str = '[FALSE]'
            if res:
                res_str = '[OK]'
            img = Draw.MolsToGridImage(ms, molsPerRow=2, subImgSize=(400, 400),
                                       legends=[cname,'mol_pred'+res_str])

            img.save('images/'+cname+'_'+str(i)+'.png')

            img.show()
            raw_input()


        if res is None:
            print("WARNING: %d %s - Could not predict from mol!"%(i,cname))
            continue
        #if i % 50 == 0:
        #logging.info("%d %r\n" % (i, res))
        if res:
            res_dict[cname]+=1

    failures=0
    corrects=0
    for key,value in res_dict.iteritems():
        print("%-12s HITS: %2d"%(key,value))
        if value==0:
            failures+=1
        else:
            corrects+=1

    print("\n%4d total " % (len(refs)))
    print("%4d found " % (len(labute_set)))
    print("%4d missed " % (missing))
    print("%4d assignment failed " % (assignment_failed))
    print("%4d unique " % (len(res_dict)))
    nall = len(labute_set)
    acc = corrects / float(nall)
    print("\nTOTAL: %5d OK: %5d WRONG: %5d Accuray: %6.3f\n" % (nall, corrects, failures, acc))


def compareDataSets(trainfile='naomi_rarey/ci300358c_si_001/naomi_viaRDKIT_train.sdf',testfile='naomi_rarey/ci300358c_si_001/naomi_viaRDKIT_test.sdf'):
    df1 = convert_sdf2dataframe(trainfile, outfile="train.csv", fillNa=np.nan, sanitize=True, tempsave=False, useSelectionRules=True, skipH=False, addBonds=True, sample=None, verbose=False)
    df2 = convert_sdf2dataframe(testfile, outfile="train.csv", fillNa=np.nan, sanitize=True, tempsave=False,
                                useSelectionRules=True, skipH=False, addBonds=True, sample=None, verbose=False)

    print df1.info()
    print df2.info()
    df1.drop(['id1','id2','bond'],inplace=True,axis=1)
    df2.drop(['id1', 'id2', 'bond'], inplace=True,axis=1)
    pcAnalysis(df1,df2)

def pcAnalysis(X, Xtest, w=None, ncomp=2, useTSNE=False):
    """
    PCA(TSNE
    """
    if useTSNE:
        print "TSNE analysis for train/test"
        pca = TSNE(n_components=ncomp)
    else:
        print "PC analysis for train/test"
        pca = TruncatedSVD(n_components=ncomp)
    print pca

    pca.fit(X)
    X_all = pd.concat([Xtest, X])
    X_r = pca.transform(X_all.values)
    plt.scatter(X_r[len(Xtest.index):, 0], X_r[len(Xtest.index):, 1], c='r', label="train", alpha=0.5)
    plt.scatter(X_r[:len(Xtest.index), 0], X_r[:len(Xtest.index), 1], c='g', label="test", alpha=0.5)
    print("Total variance:", np.sum(pca.explained_variance_ratio_))
    print("Explained variance:", pca.explained_variance_ratio_)
    plt.legend()
    plt.show()

def evaluate2d(filename='naomi_rarey/ci300358c_si_001/naomi_viaRDKIT_test.sdf',iterative=False,skipH=True):
    """
    Visually evaluate sd files

    :param filename:
    :param iterative:
    :param skipH:
    :return:
    """
    print("Evaluation run with option: noH(%r)" % (skipH))
    print("Loading classifier...")
    clf = pickle.load(open('clf.p', "rb"))
    if iterative:
        clf_iter = pickle.load(open('clf_iter.p', "rb"))
    else:
        clf_iter = None
    suppl = Chem.SDMolSupplier(filename, removeHs=False, sanitize=False)
    for i,mol in enumerate(suppl):
        print("Iteration %d"%(i))
        try:
            Chem.SanitizeMol(mol)
        except ValueError as e:
            cname = 'not_sanitized'
        cname = mol.GetProp("_Name")
        res, sdf_pred = generate_predictions(mol, skipH=skipH, iterative=iterative, forceAromatics=False, maxiter=1,
                                             verbose=True,
                                             clf=clf, clf_iter=clf_iter, isEval=True)

        if not res:
            print(Chem.MolToMolBlock(mol))
            print(sdf_pred)

        Chem.Compute2DCoords(mol)
        mol_pred = Chem.MolFromMolBlock(sdf_pred, sanitize=True)
        if mol_pred is None:
            mol_pred = Chem.MolFromMolBlock(sdf_pred, sanitize=False)
            try:
                Chem.Kekulize(mol_pred)
            except ValueError:
                print("WARNING: Could not kekulize predicted mol!")
                continue
        else:
            mol_pred = Chem.AddHs(mol_pred)
            mol_pred = Chem.RemoveHs(mol_pred)

        Chem.Compute2DCoords(mol_pred)
        ms = [mol, mol_pred]
        res_str = '[FALSE]'
        if res:
            res_str = '[OK]'


        img = Draw.MolsToGridImage(ms, molsPerRow=2, subImgSize=(400, 400),
                                   legends=[cname, 'mol_pred' + res_str])

        #img.save('images/' + cname + '_' + str(i) + '.png')
        img.show()
        raw_input()


if __name__ == "__main__":
    #clean_sdf()

    #pdb2sdf_via_templates2()
    check_cosine_law_prediction()
    #evaluate_labute()
    #compareDataSets(trainfile = 'naomi_rarey/ci300358c_si_001/naomi_viaRDKIT_train.sdf',testfile='naomi_rarey/ci300358c_si_001/naomi_viaRDKIT_test.sdf')
    #compareDataSets(trainfile='naomi_rarey/ci300358c_si_001/naomi_viaOB_train.sdf',testfile='naomi_rarey/ci300358c_si_001/naomi_viaOB_test.sdf')
    #compareDataSets(trainfile='pdbbind_refined/sdf_OB/pdbbind_refined_clean_noh_OB.sdf',testfile='pdbbind_refined/sdf/pdbbind_refined_clean_noh.sdf')
    #evaluate2d(filename='naomi_rarey/ci300358c_si_001/naomi_viaOB.sdf')




