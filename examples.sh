#!/bin/bash

#basic training & evaluation with gradient boosting
#./mamba.py --train bradley_mp_dataset_part1_UFF.sdf
#./mamba.py --eval bradley_mp_dataset_part2_UFF.sdf
#./mamba.py --train bradley_mp_dataset_part1_ETKDG.sdf
#./mamba.py --eval bradley_mp_dataset_part2_ETKDG.sdf
#./mamba.py --train bradley_mp_dataset_part1_MMFF.sdf
#./mamba.py --eval bradley_mp_dataset_part2_MMFF.sdf
#./mamba.py --train 3dqsar_sd/3dqsar_all_train.sdf
#./mamba.py --eval 3dqsar_sd/3dqsar_all_test.sdf
#./mamba.py --train gdb1/gdb1_train.sdf 
#./mamba.py --eval gdb1/gdb1_test.sdf 
#./mamba.py --train --noH pdbbind_refined/sdf/pdbbind_refined_clean_noh_train.sdf
#./mamba.py --eval --noH pdbbind_refined/sdf/pdbbind_refined_clean_noh_test.sdf

#train on subsample of all datasets
#./mamba.py --train --sample 0.3 all_train.sdf
#./mamba.py --eval bradley_mp_dataset_part2_UFF.sdf
#./mamba.py --eval bradley_mp_dataset_part2_ETKDG.sdf
#./mamba.py --eval bradley_mp_dataset_part2_MMFF.sdf
#./mamba.py --eval 3dqsar_sd/3dqsar_all_test.sdf
#./mamba.py --eval gdb1/gdb1_test.sdf 

#evaluate openbabel
#./mamba.py --evalOB --noH bradley_mp_dataset_part2_UFF.sdf
#./mamba.py --evalOB --noH bradley_mp_dataset_part2_MMFF.sdf
#./mamba.py --evalOB --noH bradley_mp_dataset_part2_ETKDG.sdf
#./mamba.py --evalOB --noH 3dqsar_all.sdf
#./mamba.py --evalOB --noH pdbbind_refined/sdf/pdbbind_refined_clean_noh_test.sdf
#./mamba.py --evalOB --noH gdb1/gdb1_test.sdf 

#use RF for training
#./mamba.py --train --useRF bradley_mp_dataset_part1_UFF.sdf
#./mamba.py --eval bradley_mp_dataset_part2_UFF.sdf
#./mamba.py --train --useRF bradley_mp_dataset_part1_ETKDG.sdf
#./mamba.py --eval bradley_mp_dataset_part2_ETKDG.sdf
#./mamba.py --train --useRF bradley_mp_dataset_part1_MMFF.sdf
#./mamba.py --eval bradley_mp_dataset_part2_MMFF.sdf
#./mamba.py --train --useRF 3dqsar_sd/3dqsar_all_train.sdf
#./mamba.py --eval 3dqsar_sd/3dqsar_all_test.sdf
#./mamba.py --train --useRF gdb1/gdb1_train.sdf 
#./mamba.py --eval gdb1/gdb1_test.sdf 
#./mamba.py --train --useRF --noH pdbbind_refined/sdf/pdbbind_refined_clean_noh_train.sdf
#./mamba.py --eval --noH pdbbind_refined/sdf/pdbbind_refined_clean_noh_test.sdf

#./mamba.py --train --useRF --sample 0.3 all_train.sdf
#./mamba.py  --eval --sample 0.3 all_train.sdf
#./mamba.py  --analyze dummy

#./mamba.py --eval bradley_mp_dataset_part2_UFF.sdf
#./mamba.py --eval bradley_mp_dataset_part2_MMFF.sdf
#./mamba.py --eval bradley_mp_dataset_part2_ETKDG.sdf
#./mamba.py --eval 3dqsar_sd/3dqsar_all_test.sdf
#./mamba.py --eval gdb1/gdb1_test.sdf 

./mamba.py --train --iterative --useRF --sample 0.3 all_train.sdf

