import os
import json

with open(os.path.join('../data/demo/trainset','search.train.json'),'r',encoding='utf-8') as f1:
    ll = [json.loads(line.strip()) for line in f1.readlines()]
    total = len(ll) // 10
    for i in range(total):
        json.dump(ll[i * 10:(i + 1) * 10],open("../data/preprocessed/new_search" + str(i) + ".json", "w", encoding='utf-8'), ensure_ascii=False,indent=True)
print("end")