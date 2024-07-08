# ROCKET: Harnessing Open-Source Knowledge for Task Augmentation

## Abstract
The success of scaling laws in large language models (LLMs) and their applications highlights the critical importance of high-quality data for cultivating emergent capabilities. The open-source community provides abundant real-world and synthetic datasets with various pre-trained LoRA models, which facilitates the rapid development of downstream applications under data-scarce scenarios. In consideration of the model's performance on one or more specific tasks, we can't help but wonder how to address the following problems: 1) how can we accurately pinpoint the existing LoRA models that benefit downstream adaptation, especially when few human-verified task-specific data are available for model evaluation and selection;
2) how can we effectively take advantage of multiple models if each of them parameterizes partial-yet-supplementary domain knowledge that is indispensable to solving tasks at hand. In this paper, we demonstrate the establishment of a scalable and generalizable pipeline for robust production of models that excel in tasks of interest. We start by maintaining our source-reliable LoRA bank with a pool of models fine-tuned on high-quality open datasets. Then, we propose to efficiently pinpoint the most potential models for each task from the pool and carefully fuse their knowledge via a mixture of experts (MoE). Finally, we propose a data selection and augmentation scheme to further improve performance on tasks of interest. Extensive experimental results confirm the superiority of our approach over existing methods in harnessing open-source knowledge.

## Model Selection

![Alt text](Figs/Model_Selection.png)

In order to run the model selection script, you need to prepare the following:

1. Base model
2. LoRA model corresponding to the Base model
3. Data with answers associated with its CoT process


```bash
### Loss Computation
cd Model_Selection
CUDA_VISIBLE_DEVICES=0 python compute_loss.py \
    --batch_size 1 \
    --model_path pretrained_models/mistral-7b-v0.2 \
    --peft_path Rocket/LoRA_Bank/mistral/arc-c_cot_lora_r16 \
    --data_path Data/arc-c_cot_50shot.json 
```

To obtain the result of K-shot evaluation, we use [opencompass](https://github.com/open-compass/opencompass). Thanks to their great work.


## Data Selection
![Alt text](Figs/Data_Selection.png)

Data selection requires preparation of: 1) alternative open-source dataset and 2) K-shot data required for the task of interest. By calculating the similarity between open-source data and K-shot data, the most relevant data are extracted, and the data with high similarity are removed.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python data_aug.py \
--opensource_path Data/commonsense_qa.json \
--kshot_data_path Data/arc-c_50shot.json \
```

## Mixture of Expert Training 
![Alt text](Figs/MoE.png)
To train the MoE model, you need to prepare the model, store it in the corresponding path, and run the following script


```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --num_gpus=4 finetune.py \
--deepspeed "./ds_config_zero3.json" \
--base_model 'pretrained_models/LLaMa-2-7B' \
--data_path 'Data_Augmentation/arc-c_aug.json' \
--output_dir './checkpoints/arc-c' \
--prompt_template_name 'llama2' \
--num_epochs 5 \
--cutoff_len 1024 \
--micro_batch_size 2 \
--load_expert_weight expert_weight/llama/arc-c.txt \
--train_router_only False \
--lora_type "lora_moe"
```

You can also download our trained MoE model directly and use [opencompass](https://github.com/open-compass/opencompass) to inference and do evaluation.