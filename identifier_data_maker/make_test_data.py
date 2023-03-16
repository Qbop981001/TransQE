import os
import numpy as np
import random

wk_dir = "../exp_data/parallel_data/wmt/en-zh/train-parts/"
files=os.listdir(wk_dir)
files = [item for item in files if '.' in item and item[0] != '.' and 'backtrans' not in item]

def random_select(file):
    zh_data, en_data = [], []
    filename_zh = file + '.zho'
    filename_en = file + '.eng'
    print(f"Reading {file}")
    with open(wk_dir+filename_zh,'r') as zh, open(wk_dir+filename_en,'r') as en:
        zh_lines = zh.readlines()
        en_lines = en.readlines()
        length = len(zh_lines)
        print(f"Select from {filename_zh}, total {length} lines.")
        indexes = np.arange(0,length)
        random.shuffle(indexes)
        selected_indexes = indexes.tolist()[:100]
        zh_data = [zh_lines[index] for index in selected_indexes]
        en_data = [en_lines[index] for index in selected_indexes]
    new_file_en = wk_dir + 'hundred_samples/' + filename_en
    new_file_zh = wk_dir + 'hundred_samples/' + filename_zh
    print(f"Write {new_file_zh}")
    with open(new_file_zh,'w') as zh, open(new_file_en,'w') as en:
        for line in zh_data:
            zh.write(line)

        for line in en_data:
            en.write(line)

for item in files:
    file = '.'.join(item.split('.')[:-1])
    random_select(file)