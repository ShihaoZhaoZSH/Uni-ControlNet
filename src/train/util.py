import random

import numpy as np


def read_anno(anno_path):
    fi = open(anno_path)
    lines = fi.readlines()
    fi.close()
    file_ids, annos = [], []
    for line in lines:
        id, txt = line.split('\t')
        file_ids.append(id)
        annos.append(txt)
    return file_ids, annos


def keep_and_drop(conditions, keep_all_prob, drop_all_prob, drop_each_prob):
    results = []
    seed = random.random()
    if seed < keep_all_prob:
        results = conditions
    elif seed < keep_all_prob + drop_all_prob:
        for condition in conditions:
            results.append(np.zeros(condition.shape))
    else:
        for i in range(len(conditions)):
            if random.random() < drop_each_prob[i]:
                results.append(np.zeros(conditions[i].shape))
            else:
                results.append(conditions[i])
    return results