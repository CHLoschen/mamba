#!/bin/bash

#basic training & evaluation with gradient boosting
#./ml_bond_parser.py --train all_sdf/bradley_mp_dataset_part1_UFF.sdf
#./ml_bond_parser.py --eval all_sdf/bradley_mp_dataset_part2_UFF.sdf
#./ml_bond_parser.py --train all_sdf/bradley_mp_dataset_part1_ETKDG.sdf
#./ml_bond_parser.py --eval all_sdf/bradley_mp_dataset_part2_ETKDG.sdf
#./ml_bond_parser.py --train all_sdf/bradley_mp_dataset_part1_MMFF.sdf
#./ml_bond_parser.py --eval all_sdf/bradley_mp_dataset_part2_MMFF.sdf
#./ml_bond_parser.py --train 3dqsar_sd/3dqsar_all_train.sdf
#./ml_bond_parser.py --eval 3dqsar_sd/3dqsar_all_test.sdf
#./ml_bond_parser.py --train gdb1/gdb1_train.sdf 
#./ml_bond_parser.py --eval gdb1/gdb1_test.sdf 
#./ml_bond_parser.py --train --noH pdbbind_refined/sdf/pdbbind_refined_clean_noh_train.sdf
#./ml_bond_parser.py --eval --noH pdbbind_refined/sdf/pdbbind_refined_clean_noh_test.sdf

#train on subsample of all datasets
#./ml_bond_parser.py --train --sample 0.3 all_sdf/train/all_train.sdf
#./ml_bond_parser.py --eval all_sdf/bradley_mp_dataset_part2_UFF.sdf
#./ml_bond_parser.py --eval all_sdf/bradley_mp_dataset_part2_ETKDG.sdf
#./ml_bond_parser.py --eval all_sdf/bradley_mp_dataset_part2_MMFF.sdf
#./ml_bond_parser.py --eval 3dqsar_sd/3dqsar_all_test.sdf
#./ml_bond_parser.py --eval gdb1/gdb1_test.sdf 

#evaluate openbabel
#./ml_bond_parser.py --evalOB --noH all_sdf/bradley_mp_dataset_part2_UFF.sdf
#./ml_bond_parser.py --evalOB --noH all_sdf/bradley_mp_dataset_part2_MMFF.sdf
#./ml_bond_parser.py --evalOB --noH all_sdf/bradley_mp_dataset_part2_ETKDG.sdf
#./ml_bond_parser.py --evalOB --noH all_sdf/3dqsar_all.sdf
#./ml_bond_parser.py --evalOB --noH pdbbind_refined/sdf/pdbbind_refined_clean_noh_test.sdf
#./ml_bond_parser.py --evalOB --noH gdb1/gdb1_test.sdf 

#use RF for training
#./ml_bond_parser.py --train --useRF all_sdf/bradley_mp_dataset_part1_UFF.sdf
#./ml_bond_parser.py --eval all_sdf/bradley_mp_dataset_part2_UFF.sdf
#./ml_bond_parser.py --train --useRF all_sdf/bradley_mp_dataset_part1_ETKDG.sdf
#./ml_bond_parser.py --eval all_sdf/bradley_mp_dataset_part2_ETKDG.sdf
#./ml_bond_parser.py --train --useRF all_sdf/bradley_mp_dataset_part1_MMFF.sdf
#./ml_bond_parser.py --eval all_sdf/bradley_mp_dataset_part2_MMFF.sdf
#./ml_bond_parser.py --train --useRF 3dqsar_sd/3dqsar_all_train.sdf
#./ml_bond_parser.py --eval 3dqsar_sd/3dqsar_all_test.sdf
#./ml_bond_parser.py --train --useRF gdb1/gdb1_train.sdf 
#./ml_bond_parser.py --eval gdb1/gdb1_test.sdf 
#./ml_bond_parser.py --train --useRF --noH pdbbind_refined/sdf/pdbbind_refined_clean_noh_train.sdf
#./ml_bond_parser.py --eval --noH pdbbind_refined/sdf/pdbbind_refined_clean_noh_test.sdf

#./ml_bond_parser.py --train --useRF --sample 0.3 all_sdf/train/all_train.sdf
#./ml_bond_parser.py  --eval --sample 0.3 all_sdf/train/all_train.sdf
#./ml_bond_parser.py  --analyze dummy

#./ml_bond_parser.py --eval all_sdf/bradley_mp_dataset_part2_UFF.sdf
#./ml_bond_parser.py --eval all_sdf/bradley_mp_dataset_part2_MMFF.sdf
#./ml_bond_parser.py --eval all_sdf/bradley_mp_dataset_part2_ETKDG.sdf
#./ml_bond_parser.py --eval 3dqsar_sd/3dqsar_all_test.sdf
#./ml_bond_parser.py --eval gdb1/gdb1_test.sdf 

./ml_bond_parser.py --train --iterative --useRF --sample 0.3 all_sdf/train/all_train.sdf

