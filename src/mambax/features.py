"""Feature extraction from molecular structures."""

import logging
import os
import sys
from typing import Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit.Chem import AllChem as Chem
from scipy.spatial.distance import pdist, squareform

from .io import convert_smiles2sdfile, read_pdbfile, read_xyz


def extract_features(
    mol: Optional[Any],
    sourcename: str,
    pos: int,
    printHeader: bool = True,
    fillNa: float = np.nan,
    xyz_file: Optional[str] = None,
    plot: bool = False,
    useSelectionRules: bool = True,
    OrderAtoms: bool = True,
    bondAngles: bool = True,
    skipH: bool = False,
    addBonds: bool = False,
    verbose: bool = False
) -> Optional[pd.DataFrame]:
    """Extract feature matrix from RDKit mol object or xyz file."""

    pt = Chem.GetPeriodicTable()

    if xyz_file is not None:
        if xyz_file.lower().endswith(".xyz"):
            atomtypes, coords, q, _ = read_xyz(xyz_file, skipH=skipH)
        elif xyz_file.lower().endswith(".pdb"):
            atomtypes, coords, q, _ = read_pdbfile(xyz_file, skipH=skipH)
        if q!=0:
            logging.info("Found charge: %.2f"%(q))
        dm = squareform(pdist(np.asarray(coords)))
    else:
        if skipH:
            try:
                mol = Chem.RemoveHs(mol)
            except ValueError:
                logging.info("Skipping H deletion for molecule at pos:" + str(pos))
                return None
        #check if bonds are available
        try:
            if not addBonds and mol.GetNumBonds(onlyHeavy=False)==0:
                logging.info("No bonds found: skipping molecule %s " %Chem.MolToSmiles(mol))
                return None
        except RuntimeError:
            logging.info("RuntimeError: skipping molecule")
            return None

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
        for i in range(2*n_cut):
            if addBonds:
                print('{:>4s}{:>3s}{:>8s}{:>8s}{:>4s}'.format('POS', '#', 'DIST', 'DISTB','BNB'),end='')
            elif bondAngles:
                print('{:>4s}{:>3s}{:>8s}{:>8s}'.format('POS', '#', 'DIST','DISTB'),end='')
            else:
                print('{:4s}{:3s}{:8s}'.format('POS', '#', 'DIST'),end='')
        print("{:4s}".format('TYPE'))

    df = []
    index = []

    for i in range(0,n):
        if xyz_file is not None:
            bnd_at1 = atomtypes[i]
            bond_num1 = pt.GetAtomicNumber(bnd_at1)
        else:
            bnd_at1 = mol.GetAtomWithIdx(i)
            bond_num1 = bnd_at1.GetAtomicNum()
            bnd_at1 = bnd_at1.GetSymbol()

        for j in range(0,m):
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
                    logging.warning("Unreasonable short X-X bond (r<0.75): %r(%d) %r(%d) %4.2f type: %d from source: %s at pos: %d"%(bnd_at1,i+1,bnd_at2,j+1,bnd_dist,bnd_type,sourcename,pos))
                elif bnd_dist<1.1 and bond_num1>=6 and bond_num2>=6 and bnd_type>0:
                    logging.warning("Unreasonable short X-X bond (r<1.1): %r(%d) %r(%d) %4.2f type: %d from source: %s at pos: %d"%(bnd_at1,i+1,bnd_at2,j+1,bnd_dist,bnd_type,sourcename,pos))
                # in case of problems we discard whole molecule
                elif bnd_dist < 0.75 and (bond_num1 == 1 or bond_num2 == 1) and bnd_type == 0:
                    logging.warning("%s unreasonable short X-H distance w/o bond: %r(%d) %r(%d) %4.2f type: %d from source: %s at pos: %d" % (selstr,
                        bnd_at1,i+1, bnd_at2,j+1, bnd_dist, bnd_type,sourcename,pos))
                    if useSelectionRules: return None
                elif bnd_dist < 1.5 and bond_num1==6 and bond_num2==6 and bnd_type==0:
                    logging.warning("%s unreasonable short C-C distance w/o bond: %r(%d) %r(%d) %4.2f type: %d from source: %s at pos: %d" % (selstr,
                    bnd_at1,i+1, bnd_at2,j+1, bnd_dist, bnd_type,sourcename,pos))
                    if useSelectionRules: return None
                elif bnd_dist < 1.0 and bond_num1>=6 and bond_num2>=6 and bnd_type==0:
                    logging.warning("%s unreasonable short distance w/o bond: %r(%d) %r(%d) %4.2f type: %d from source: %s at pos: %d" % (selstr,
                    bnd_at1,i+1, bnd_at2,j+1, bnd_dist, bnd_type,sourcename,pos))
                    if useSelectionRules: return None
                # rather generous cutoff
                elif bnd_dist>1.8 and bond_num1==6 and bond_num2==6 and bnd_type>0:
                    logging.warning("%s unreasonable long C-C bond: %r(%d) %r(%d) %4.2f type: %d from source: %s at pos: %d"%(selstr,bnd_at1,i+1,bnd_at2,j+1,bnd_dist,bnd_type,sourcename,pos))
                    if useSelectionRules: return None

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
                        nextn = int(nextn)
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
        # i.e. for empty dataframes
        df = None
    return df


def convert_sdf2dataframe(
    infile: str,
    outfile: str = "moldat.csv",
    fillNa: float = np.nan,
    sanitize: bool = True,
    tempsave: bool = False,
    useSelectionRules: bool = True,
    skipH: bool = False,
    addBonds: bool = True,
    sample: Optional[float] = None,
    debug: bool = False,
    verbose: bool = False
) -> Optional[pd.DataFrame]:
    """Generate training dataset from SDF files."""
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
                Chem.SanitizeMol(mol)
            except ValueError:
                logging.info("Skipping sanitization for molecule at pos:" + str(i+1))
                if debug:
                    w = Chem.SDWriter('tmp_pos'+str(i+1)+'.sdf')
                    w.write(mol)
                    w.close()

        if mol is not None:
            if sample is not None and np.random.random_sample()>sample:
                continue
            df_features = extract_features(mol, infile, i, verbose=verbose, printHeader=(i==0), fillNa=fillNa, useSelectionRules=useSelectionRules, skipH=skipH, addBonds=addBonds)
            if df_features is not None:
                if df_new is None:
                    df_new = df_features
                else:
                    df_new = pd.concat([df_new, df_features], axis=0)
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

    if df_new is not None:
        logging.info("Bond types: \n%r"%(df_new['bond'].value_counts()))
        logging.info("Total bonds: %r\n" % (df_new['bond'].value_counts().sum()))
    return df_new


def convert_sdfiles2csv(
    file_list: List[str] = [],
    base_dir: str = '',
    outdat: str = 'train_dat.csv',
    method: str = 'UFF',
    skipH: bool = False,
    addBonds: bool = False,
    sample: float = 0.25,
    verbose: bool = False
) -> None:
    """Convert list of SDF files to CSV training data."""
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

