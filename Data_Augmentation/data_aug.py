import torch
from tqdm import tqdm
import numpy as np  
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
import json
from scipy.spatial.distance import cdist

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
    close_points = np.where(dists < threshold)
    to_remove = []
    for i in range(len(close_points[0])):
        if close_points[0][i] != close_points[1][i] and close_points[1][i] not in to_remove:
            to_remove.append(close_points[1][i])

    remaining_indices = np.delete(np.arange(data.shape[0]), to_remove)
    return remaining_indices

def main():
    split_num = 1024
    target_num = 5000
    prefix = "Represent the following sentence for similar task retrieval: "
    text_list = []
    with open("/apdcephfs_cq8/share_2992827/shennong_5/yunchengyang/Rocket/Data/commonsense_qa.json", 'r') as f:
        data_bank = f.readlines()
        data = []
        for i, d in enumerate(data_bank):
            item = json.loads(d)
            text_list.append(prefix+"instruction:"+item["instruction"]+'\n'+"input:"+item["input"]+'\n'+"output:"+item["output"])
            data.append(item)

    probe_text_list = []
    with open("/apdcephfs_cq8/share_2992827/shennong_5/yunchengyang/Rocket/Data/arc-c_50shot.json", 'r') as f:
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


    dataset = CorpusDataset(text_list, tokenizer, max_length=split_num)
    dataloader = DataLoader(dataset, batch_size=400, shuffle=False, num_workers=8)
    tbar = tqdm(dataloader)

    probe_dataset = CorpusDataset(probe_text_list, tokenizer, max_length=split_num)
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
    closest_indices = np.argpartition(average_distances, selected_num)[:selected_num]
    closest_embeddings = embeddings[closest_indices]


    data = np.array(data)
    selected_data = data[closest_indices]
    div_idx = remove_close_points(closest_embeddings, 0.1)
    div_data = selected_data[div_idx].tolist()

    with open("arc-c_aug.json", 'w') as f:
        for data in div_data:
            f.write(json.dumps(data)+'\n')

if __name__ == '__main__':
    main()
