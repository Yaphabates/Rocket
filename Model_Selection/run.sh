CUDA_VISIBLE_DEVICES=0 python compute_ppl.py \
    --batch_size 1 \
    --model_path pretrained_models/mistral-7b-v0.2 \
    --peft_path Rocket/LoRA_Bank/mistral/arc-c_cot_lora_r16 \
    --data_path Rocket/Data/arc-c_cot_50shot.json 