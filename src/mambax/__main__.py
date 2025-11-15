#!/usr/bin/env python
"""CLI entry point for MAMBAX."""

from time import time
import sys

import numpy as np
import pandas as pd

from .evaluation import eval_job, evaluate, evaluate_OB
from .ml import generate_multi_predictions, generate_predictions, train_job
from .utils import (
    clean_sdf,
    get_stats,
    sanitize_sdf,
    set_logging,
    shuffle_split_sdfile,
)


def main():
    """Main CLI entry point."""
    import argparse

    t0 = time()
    pd.set_option('display.max_columns', 30)
    pd.set_option('display.width', 1000)

    np.random.seed(42)
    set_logging()

    description = """
    MAMBAX - Machine Learning Meets Bond Analytix
       
    Creates .sdf file incl. chemical bond information out of a .xyz file
    Bond perception is learned via machine learning (scikit-learn classifier) from
    previous .sdf or .smi files.
    
    Internally uses RDKit, numpy, pandas and scikit-learn python packages
    
    (c) 2018 Christoph Loschen
    
    Examples:
    
    1) mambax --train largefile.sdf
    2) mambax --add special_case.sdf [optional]
    3) mambax --predict new_molecule.xyz
    
    
    """
  
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)

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
    group.add_argument('--shufflesplit', type=float, metavar='f', help='Create test and train set by splitting and shuffling molecules, f is a float [0..1] \n')

    parser.add_argument('--noH', action='store_true', help='omit Hydrogen atom when learning\n', default=False)
    parser.add_argument('--useRF', action='store_true', help='use Random Forest instead of Gradient Boosting for training\n', default=False)
    parser.add_argument('-v','--verbose', action='store_true', help='verbose\n', default=False)
    parser.add_argument('--iterative', action='store_true', help='Iterative prediction of bonds using 2 classifiers\n', default=False)
    parser.add_argument('--sample', type=float, default=None, metavar='f', help='Subsampling of dataset during training, f is a float [0..1] \n')
    parser.add_argument('--FF', choices=("UFF", "MMFF", "ETKDG"), help='Forcefield/Method to use for 3D structure generation from SMILES', required=False,default="UFF")

    parser.add_argument("filename", nargs='+', help=".sdf or .smi for training, .xyz for prediction")

    fmethod = 'UFF'

    args = parser.parse_args()
    if args.FF!='UFF':
        fmethod=args.FF

    if args.verbose:
        print("Verbose ON")

    if len(args.filename) > 1:
        if args.predict and all(f.lower().endswith((".xyz", ".pdb")) for f in args.filename):
            generate_multi_predictions(args.filename, iterative=args.iterative, skipH=args.noH, verbose=args.verbose)
        else:
            print("Multiple files only allowed for prediction with .xyz and .pdb files")
            sys.exit(1)
    else:
        for f in args.filename:

            if args.train:
                if args.sample is not None and not 0.0 <= args.sample <= 1.0:
                    sys.stderr.write("ERROR: --sample fraction must be between 0.0 and 1.0\n")
                    sys.exit(1)
                train_job(f, reset=True, fmethod=fmethod, skipH=args.noH, iterative=args.iterative, useRF=args.useRF, sample=args.sample)

            elif args.add:
                if args.sample is not None and not 0.0 <= args.sample <= 1.0:
                    sys.stderr.write("ERROR: --sample fraction must be between 0.0 and 1.0\n")
                    sys.exit(1)
                train_job(f, reset=False, fmethod=fmethod, skipH=args.noH, iterative=args.iterative, useRF=args.useRF, sample=args.sample)

            elif args.predict:
                generate_predictions(f, iterative=args.iterative,writeFile=True, verbose=args.verbose)

            elif args.eval and args.iterative:
                eval_job(f, skipH=args.noH, iterative=args.iterative,verbose=args.verbose)

            elif args.eval:
                if args.sample is not None and not 0.0 <= args.sample <= 1.0:
                    sys.stderr.write("ERROR: --sample fraction must be between 0.0 and 1.0\n")
                    sys.exit(1)
                train_job(f, eval=True, fmethod=fmethod, skipH=args.noH, iterative=args.iterative, sample=args.sample, verbose=args.verbose)

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

            elif args.shufflesplit is not None:
                if not 0.0 <= args.shufflesplit <= 1.0:
                    sys.stderr.write("ERROR: --shufflesplit fraction must be between 0.0 and 1.0\n")
                    sys.exit(1)
                shuffle_split_sdfile(f, frac=args.shufflesplit)

    print("FINISHED in %fs\n" % (time() - t0))


if __name__ == "__main__":
    main()

