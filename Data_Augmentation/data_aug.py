import torch
from tqdm import tqdm
import numpy as np  
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
import json
from scipy.spatial.distance import cdist
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kshot_data_path', type=str, default="")
    parser.add_argument('--opensource_path', type=str, default="")
    parser.add_argument('--target_num', type=int, default=1000)
    parser.add_argument('--cutoff_len', type=int, default=1024)
    args = parser.parse_args()
    return args

class CorpusDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, index):
        text = self.texts[index]
        text = self.tokenizer([text], return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
        input_ids = text["input_ids"][0]
        attention_mask = text["attention_mask"][0]
        return input_ids,attention_mask

    def __len__(self):
        return len(self.texts)

def remove_close_points(data, threshold):
    # cosine distance
    dists = cdist(data, data, 'cosine')
    to_remove = []

    for i in range(len(data)-1, -1, -1):
        for j in range(i-1, -1, -1):
            if dists[i][j] < threshold:
                to_remove.append(i)
                break
            
    remaining_indices = np.delete(np.arange(data.shape[0]), to_remove)

    return remaining_indices

def main(args):
    cutoff_len = args.cutoff_len
    target_num = args.target_num

    prefix = "Represent the following sentence for similar task retrieval: "
    text_list = []
    with open(args.opensource_path, 'r') as f:
        data_bank = f.readlines()
        data = []
        for i, d in enumerate(data_bank):
            item = json.loads(d)
            text_list.append(prefix+"instruction:"+item["instruction"]+'\n'+"input:"+item["input"]+'\n'+"output:"+item["output"])
            data.append(item)

    probe_text_list = []
    with open(args.kshot_data_path, 'r') as f:
        probe_data = f.readlines()
        for pd in probe_data:
            item = json.loads(pd)
            probe_text_list.append(prefix+"instruction:"+item["instruction"]+'\n'+"input:"+item["input"]+'\n'+"output:"+item["output"])
    

    selected_num = target_num - len(probe_text_list)

    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')
    model = AutoModel.from_pretrained('BAAI/bge-m3', torch_dtype=torch.float16)
    if torch.cuda.is_available():
        available_gpus =[i for i in range(torch.cuda.device_count())]
        if len(available_gpus)== 1:
            model = model.cuda()
        else:
            model = torch.nn.DataParallel(model, device_ids=available_gpus).cuda()
    
    model.eval()


    dataset = CorpusDataset(text_list, tokenizer, max_length=cutoff_len)
    dataloader = DataLoader(dataset, batch_size=400, shuffle=False, num_workers=8)
    tbar = tqdm(dataloader)

    probe_dataset = CorpusDataset(probe_text_list, tokenizer, max_length=cutoff_len)
    probe_dataloader = DataLoader(probe_dataset, batch_size=8, shuffle=False, num_workers=8)
    probe_tbar = tqdm(probe_dataloader)

    embeddings = []
    probe_embeddings = []
    with torch.no_grad():
        for sample in tbar:
            texts = {'input_ids':sample[0], 'attention_mask': sample[1]}
            model_output = model(**texts)
            model_output = model_output[0][:, 0]
            model_output /= model_output.norm(dim=-1, keepdim=True)
            embeddings.append(model_output.cpu().numpy())
    embeddings = np.concatenate(embeddings, axis=0)

    with torch.no_grad():
        for sample in probe_tbar:
            texts = {'input_ids':sample[0], 'attention_mask': sample[1]}
            model_output = model(**texts)
            model_output = model_output[0][:, 0]
            model_output /= model_output.norm(dim=-1, keepdim=True)
            probe_embeddings.append(model_output.cpu().numpy())
    probe_embeddings = np.concatenate(probe_embeddings, axis=0)
    probe_embeddings = probe_embeddings.mean(axis=0, keepdims=True)

    distance_matrix = cdist(probe_embeddings, embeddings, metric='euclidean')
    average_distances = distance_matrix.mean(axis=0)
    # import pdb; pdb.set_trace()
    closest_indices = np.argpartition(average_distances, selected_num)[:selected_num]
    # closest_indices.sort()
    closest_embeddings = embeddings[closest_indices]


    data = np.array(data)
    selected_data = data[closest_indices]

    div_idx = remove_close_points(closest_embeddings, 0.01)
    div_data = selected_data[div_idx].tolist()

    with open("arc-c_aug.json", 'w') as f:
        for data in div_data:
            f.write(json.dumps(data)+'\n')

if __name__ == '__main__':
    args = parse_args()
    main(args)
