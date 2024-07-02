CUDA_VISIBLE_DEVICES=0 python compute_loss.py \
    --batch_size 1 \
    --model_path /apdcephfs_cq8/share_2992827/shennong_5/ianxxu/pretrained_models/mistral-7b-v0.2 \
    --peft_path /apdcephfs_cq8/share_2992827/shennong_5/yunchengyang/Rocket/LoRA_Bank/mistral/arc-c_cot_lora_r16 \
    --data_path /apdcephfs_cq8/share_2992827/shennong_5/yunchengyang/Rocket/Data/arc-c_cot_50shot.json 