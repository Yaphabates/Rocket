CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --num_gpus=4 finetune.py \
--deepspeed "./ds_config_zero3.json" \
--base_model '/apdcephfs_cq8/share_2992827/shennong_5/ianxxu/pretrained_models/LLaMa-2-7B' \
--data_path '/apdcephfs_cq8/share_2992827/shennong_5/yunchengyang/Rocket/Data/arc-c_50shot.json' \
--output_dir './checkpoints/arc-c' \
--prompt_template_name 'llama2' \
--num_epochs 10 \
--cutoff_len 1024 \
--micro_batch_size 2