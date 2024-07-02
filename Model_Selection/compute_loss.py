import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import json
import torch
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--model_path', type=str, default="")
    parser.add_argument('--peft_path', type=str, default="")
    parser.add_argument('--data_path', type=str, default="")
    args = parser.parse_args()
    return args


def generate(args):

    start_dirs = [args.data_path]
    output_dirs = ["result.json"]
    sys_prompt = "[INST] <<SYS>>\nYou are a helpful assistant. Below is an instruction that describes a task. \nWrite a response that appropriately completes the request.\n<</SYS>>\n\n {instruction}\n{input}\n [/INST]"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    from peft import PeftModel
    if args.peft_path:
        model = PeftModel.from_pretrained(model, args.peft_path, is_trainable=True)

    for i, data_path in enumerate(start_dirs):

        with open(data_path) as f:
            D = f.readlines()
            
            out_loss = []
            batch_num = 0
            items = []

            for item in D:
                items.append(item)
                qa_item = json.loads(item)
                input_ids = tokenizer.encode(sys_prompt.format(instruction=qa_item['instruction'], input=qa_item['input']))
                output_ids = tokenizer.encode(qa_item['output'])
                output_ids.append(tokenizer.eos_token_id)
                src_len = len(input_ids)
                inputs_ids = input_ids + output_ids
                inputs_len = len(inputs_ids)
                attention_mask = [1] * inputs_len
                labels = [-100] * src_len + output_ids
                  
                batch_dict = {
                    "input_ids": torch.tensor(inputs_ids).unsqueeze(0),
                    "attention_mask": torch.tensor(attention_mask).unsqueeze(0),
                    "labels": torch.tensor(labels).unsqueeze(0),
                }
                
                batch = {k: v.cuda() for k, v in batch_dict.items()}
                outputs = model(**batch)
                loss = outputs.loss
                out_loss.append(loss.item())
                
                batch_num += 1

        with open(output_dirs[i], 'w') as fw:
            for i in range(len(out_loss)):
                data_dict = {
                    'conversation': items[i],
                    'loss': out_loss[i]
                }
                fw.write(json.dumps(data_dict, ensure_ascii=False)+'\n')
            print(args.peft_path, np.mean(out_loss))


            
if __name__ == '__main__':
    args = parse_args()
    generate(args)

