import os
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset
from src.peft.utils.config import TaskType 
from src.peft.tuners.mmoelora import MMOELoraConfig
from src.utils.prompter import Prompter

from src.peft import (
    LoraConfig, 
    get_peft_model,
    set_peft_model_state_dict,
    load_expert_weights
)
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


def train(
    # model/data params
    base_model: str = "",  # the only required argument
    data_path: str = "yahma/alpaca-cleaned",
    output_dir: str = "./",
    # training hyperparams
    batch_size: int = 64,
    micro_batch_size: int = 4,
    num_epochs: float = 6,
    learning_rate: float = 5e-5,
    cutoff_len: int = 1024,
    val_set_size: int = 0,
    bf16: bool = False,
    # lora hyperparams
    lora_type: str = 'single_lora',
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    load_expert_weight: str = "",
    train_router_only: bool = False,
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "llama2",  # The prompt template to use, will default to alpaca.
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, TOKENIZERS_PARALLELISM=False)
    
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  
        return tokenized_full_prompt

    if lora_type == 'single_lora':

        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
    elif lora_type == 'lora_moe':
        kwargs = {"expert_num": 4}
        task_type = TaskType.CAUSAL_LMS
        config = MMOELoraConfig(
                task_type=task_type,
                target_modules=lora_target_modules,
                inference_mode=False,
                r=lora_r, lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                **kwargs
            )
        if load_expert_weight:
            with open(load_expert_weight, 'r') as f:
                file = f.readlines()
                expert_weight_path = [path.strip('\n') for path in file]
                expert_num = len(expert_weight_path)
                kwargs = {
                "expert_num": expert_num
                }
            config = MMOELoraConfig(
                    task_type=task_type,
                    target_modules=lora_target_modules,
                    inference_mode=False,
                    r=lora_r, lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    **kwargs
                )

    model = get_peft_model(model, config)
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)
    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    if load_expert_weight:
        with open(load_expert_weight, 'r') as f:
            file = f.readlines()
            expert_weight_path = [path.strip('\n') for path in file]
        load_expert_weights(model, expert_weight_path)

    if train_router_only:
        for name, param in model.named_parameters():
            if 'lora_gate' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
        
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        print("Data length before filter:", len(train_data))
        train_data = train_data.filter(lambda x: len(x['input_ids'])<cutoff_len)
        print("Data length after filter:", len(train_data))
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        print("Use model parallel")
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Seq2SeqTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.Seq2SeqTrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=5,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True if bf16 else False,
            fp16=True if not bf16 else False,
            logging_steps=10,
            save_safetensors=False,
            optim="adamw_torch",
            evaluation_strategy="no" if val_set_size > 0 else "no",
            save_strategy="steps",
            save_steps=1000,
            output_dir=output_dir,
            save_total_limit=3,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to=None
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False


    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir, safe_serialization=False)



if __name__ == "__main__":
    import os
    os.environ["WANDB_DISABLED"] = "true"
    fire.Fire(train)
