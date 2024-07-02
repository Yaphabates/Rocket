import torch
import torch.nn.functional as F

import itertools

expert_weight_path = "/apdcephfs_cq8/share_2992827/shennong_5/yunchengyang/MOELoRA-peft/expert_weight/arc-c.txt"

with open(expert_weight_path, 'r') as f:
    file = f.readlines()
    model_dir = [path.strip('\n') for path in file]

def weight_sim(weight_1, weight_2):
    res = 0
    for key in weight_1.keys():
        res += abs(F.cosine_similarity(weight_1[key].reshape(1, -1), weight_2[key].reshape(1, -1), eps=1e-8))
    return res

def similarity(w1, w2):
    state_dict_A = torch.load(w1, map_location='cpu')
    state_dict_B = torch.load(w2, map_location='cpu')
    diff = weight_sim(state_dict_A, state_dict_B)
    return diff

import time
start_time = time.time()


similarity_sums = []
combinations = list(itertools.combinations(model_dir, 4))

for combination in combinations:
    similarity_sum = 0
    for pair in itertools.combinations(combination, 2):
        similarity_sum += similarity(pair[0], pair[1])
    
    similarity_sums.append((similarity_sum, combination))

similarity_sums.sort()

k = 3 
for i in range(k):
    print(f"Rank {i + 1}:")
    print("Combination:", similarity_sums[i][1])
    print("Group Diversity:", similarity_sums[i][0])


end_time = time.time()
print("time", end_time-start_time)



