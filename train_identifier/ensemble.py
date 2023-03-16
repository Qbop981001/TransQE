import torch
import pandas as pd
import logging
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import time
import sys
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_recall_fscore_support

from transformers import BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
ckpt_dir = "checkpoint_trans"

DATA_DIR='../identifier_data/'
EPOCH = 2
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class RawDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings.input_ids)


def read_file(data_type,lang):

    filename1 = DATA_DIR + data_type + '.' + 'tran'
    filename2 = DATA_DIR + data_type + '.' + 'orig'
    src1 = filename1 +'.'+'src'
    src2 = filename2 +'.'+'src'
    Xs, ys = [], []
    with open(filename1) as f, open(src1) as sf: # source-original
        tgts = f.readlines()
        srcs = sf.readlines()
        assert(len(tgts)==len(srcs))
        for s, t in zip(srcs,tgts):
            if len(t) <= 128:
                if lang == 'src':
                    Xs.append(s.strip())
                    ys.append(1)
                elif lang == 'tgt':
                    Xs.append(t.strip())
                    ys.append(1)

    with open(filename2) as f, open(src2) as sf:  # target-original
        tgts = f.readlines()
        srcs = sf.readlines()
        assert(len(tgts)==len(srcs))
        for s, t in zip(srcs,tgts):
            if len(t) <= 128:
                if lang == 'src':
                    Xs.append(s.strip())
                    ys.append(0)
                elif lang == 'tgt':
                    Xs.append(t.strip())
                    ys.append(0)


    return Xs, ys


def build_dataloader(data_type,lang):


    if data_type[:5] == 'train':
        X, y = read_file(data_type,lang)
        mid = 990140
        # print(y[mid - 50:mid + 50])
        # X = X[mid-100:mid+100]
        # y = y[mid - 100:mid + 100]
        X = X[:mid-1000] + X[mid+1000:]
        y = y[:mid-1000] + y[mid+1000:]
    elif data_type[:5] == 'valid':
        X, y = read_file('train-full',lang)
        mid = 990140
        # print(y[mid - 50:mid + 50])
        # X = X[mid-11000:mid-10000] + X[mid+10000:mid+11000]
        #
        # y = y[mid-11000:mid-10000] + y[mid+10000:mid+11000]
        X = X[mid-1000:mid+1000]
        y = y[mid - 1000:mid + 1000]
    else:
        X, y = read_file(data_type,lang)
    print(f"{len(X),len(y)}")
    if lang=='src':
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    elif lang=='tgt':
        tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    print("Start tokenize")
    start = time.time()
    encodings = tokenizer(list(X), truncation=True, padding=True)
    print(f"Tokenizing using {time.time()-start}s. ")
    dataset = Dataset(encodings, y)

    if data_type == 'train' or data_type == 'train-large' or data_type == 'train-filter' or data_type == 'train-bad' or data_type == 'train-full':
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

        return train_loader
    elif data_type == 'valid':
        valid_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        return valid_loader
    elif data_type[:4] == 'test':
        test_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        return test_loader
    else:
        print('Unknown data_type')



def predict(src_loader,tgt_loader):

    model_en = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model_en.to(device)
    model_en.load_state_dict(torch.load(os.path.join(ckpt_dir, f"ft-bert-en-enzh.pth"),map_location=device))
    model_en.eval()
    model_zh = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2)
    model_zh.to(device)
    model_zh.load_state_dict(torch.load(os.path.join(ckpt_dir, f"ft-bert-zh-enzh.pth"),map_location=device))
    model_zh.eval()

    preds = []
    truths = []
    with torch.no_grad():
        for bs,bt in zip(src_loader,tgt_loader):
            input_ids = bs['input_ids'].to(device)
            attention_mask = bs['attention_mask'].to(device)
            pred_en = model_en(input_ids, attention_mask=attention_mask)[0]
            input_ids = bt['input_ids'].to(device)
            attention_mask = bt['attention_mask'].to(device)
            pred_zh = model_zh(input_ids, attention_mask=attention_mask)[0]
            pred = (pred_en + pred_zh) / 2
            # pred = pred_en
            pred = torch.argmax(pred,dim=1).detach().cpu().numpy().tolist()
            truth=bs['labels'].squeeze().cpu().numpy().tolist()
            # assert(bs['labels']==bt['labels'])
            if type(truth) == type(0):
                truth = [truth]
            # print(truth)
            preds.extend(pred)
            truths.extend(truth)


        return precision_recall_fscore_support(truths,preds, average='macro', labels=[0,1])


def build_paraloader(filename):
    with open(filename+'.zh') as f, open(filename+'.en') as sf: # source-original
        tgts = f.readlines()
        srcs = sf.readlines()
        assert(len(tgts)==len(srcs))
        for i in range(len(tgts)):
            tgts[i] = ''.join(tgts[i].strip().split())
    srcs=srcs
    tgts=tgts
    start = time.time()
    tokenizer_en = BertTokenizer.from_pretrained("bert-base-uncased")
    encodings_en = tokenizer_en(list(srcs), truncation=True, padding=True)
    print(f"finished tokenizing english, using {time.time()-start}s")
    tokenizer_zh = BertTokenizer.from_pretrained("bert-base-chinese")
    encodings_zh = tokenizer_zh(list(tgts), truncation=True, padding=True)
    print(f"finished, using {time.time()-start}s")
    dataset_en = RawDataset(encodings_en)
    dataset_zh = RawDataset(encodings_zh)

    src_loader = DataLoader(dataset_en, batch_size=8, shuffle=False)
    tgt_loader = DataLoader(dataset_zh, batch_size=8, shuffle=False)
    print(len(srcs),len(tgts))
    return src_loader, tgt_loader


def predict_parallel(filename):
    src_loader,tgt_loader = build_paraloader(filename)
    model_en = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model_en.to(device)
    model_en.load_state_dict(torch.load(os.path.join(ckpt_dir, f"ft-bert-en_enzh.pth"),map_location=device))
    model_en.eval()
    model_zh = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2)
    model_zh.to(device)
    model_zh.load_state_dict(torch.load(os.path.join(ckpt_dir, f"ft-bert-zh_enzh.pth"),map_location=device))
    model_zh.eval()


    preds = []
    zh_embedding, en_embedding = [], []
    print("Start identification of parallel_data data")
    with torch.no_grad():
        for bs,bt in tqdm(zip(src_loader,tgt_loader)):
            input_ids = bs['input_ids'].to(device)
            attention_mask = bs['attention_mask'].to(device)
            pred_en,hidden_states = model_en(input_ids, attention_mask=attention_mask,output_hidden_states=True)
            cls_last_hidden_states = hidden_states[-1][:, 0, :].cpu().numpy().tolist()
            en_embedding.extend(cls_last_hidden_states)
            input_ids = bt['input_ids'].to(device)
            attention_mask = bt['attention_mask'].to(device)
            pred_zh, hidden_states = model_zh(input_ids, attention_mask=attention_mask,output_hidden_states=True)
            pred = (pred_en + pred_zh) / 2
            cls_last_hidden_states=hidden_states[-1][:,0,:].cpu().numpy().tolist()
            zh_embedding.extend(cls_last_hidden_states)

            pred = torch.argmax(pred,dim=1).detach().cpu().numpy().tolist()
            preds.extend(pred)
            if len(preds) % 10000 == 0:
                print(len(preds))


    print(len(preds))
    print(sum([1 for item in preds if item==1]))
    with open(sys.argv[3],'w') as f:
        for item in preds:
            f.write(str(item)+'\n')
    with open(sys.argv[4]+'.zh','w') as f:
        for embed in zh_embedding:
            f.write(' '.join([str(item) for item in embed])+'\n')

    with open(sys.argv[4]+'.en','w') as f:
        for embed in en_embedding:
            f.write(' '.join([str(item) for item in embed])+'\n')
    # preds = []
    # print("Start identification of parallel_data data, using ChineseBert only.")
    # with torch.no_grad():
    #     for bt in tgt_loader:
    #
    #         input_ids = bt['input_ids'].to(device)
    #         attention_mask = bt['attention_mask'].to(device)
    #         pred_zh = model_zh(input_ids, attention_mask=attention_mask)[0]
    #         pred =  pred_zh
    #         pred = torch.argmax(pred,dim=1).detach().cpu().numpy().tolist()
    #         preds.extend(pred)
    #
    #
    # print(len(preds))
    # print(sum([1 for item in preds if item==1]))
    # with open("zh_only_identification_results.txt",'w') as f:
    #     for item in preds:
    #         f.write(str(item)+'\n')


if __name__ == '__main__':

    if sys.argv[1] == 'predict':
        for i in range(21,22):
            src_loader = build_dataloader(f"test{i}",'src')
            tgt_loader = build_dataloader(f"test{i}",'tgt')
            out = predict(src_loader,tgt_loader)

            print(f"S-O T-O identification in test{i} : {out[0]},{out[1]},{out[2]}\n=============end============")

    elif sys.argv[1] == 'predict-para':
        predict_parallel(sys.argv[2])

    elif sys.argv[1] == 'predict-sample':
        pass

