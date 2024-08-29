CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python data_aug.py \
--opensource_path ../Data/opensouece-data.json \
--kshot_data_path arc_boolq_piqa.json \
--target_num 5000