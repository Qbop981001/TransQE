import torch
import pandas as pd
from transformers import XLMRobertaTokenizer
from torch.utils.data import DataLoader

import os
import time
import sys
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_recall_fscore_support

from transformers import XLMRobertaForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
ckpt_dir = "checkpoint_trans"
if not os.path.exists(ckpt_dir):
    os.mkdir(ckpt_dir)
    print("Checkpoint directory established.")
# with open('sentiment_classification_result.txt','w') as f:
#     f.write("Evaluation of emotion extraction.")

DATA_DIR='/workspace/translationese/data/'
EPOCH = 3
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


def read_file(data_type):

    filename1 = DATA_DIR + data_type + '.' + 'tran'
    filename2 = DATA_DIR + data_type + '.' + 'orig'
    src1 = filename1 +'.'+'src'
    src2 = filename2 +'.'+'src'
    Xs, ys = [], []
    with open(filename1) as f, open(src1) as sf:
        tgts = f.readlines()
        srcs = sf.readlines()
        assert(len(tgts)==len(srcs))
        for s, t in zip(srcs,tgts):
            if len(t) <= 128:
                Xs.append(s.strip()+'</s>'+t.strip())
                ys.append(1)

    with open(filename2) as f, open(src2) as sf:
        tgts = f.readlines()
        srcs = sf.readlines()
        assert(len(tgts)==len(srcs))
        for s, t in zip(srcs,tgts):
            if len(t) <= 128:
                Xs.append(s.strip()+'</s>'+t.strip())
                ys.append(0)


    return Xs, ys


def build_dataloader(data_type):


    if data_type[:5] == 'train':
        X, y = read_file(data_type)
        mid = 990140
        # print(y[mid - 50:mid + 50])
        # X = X[mid-10000:mid+10000]
        # y = y[mid - 10000:mid + 10000]
        X = X[:mid-1000] + X[mid+1000:]
        y = y[:mid-1000] + y[mid+1000:]
    elif data_type[:5] == 'valid':
        X, y = read_file('train-full')
        mid = 990140
        # print(y[mid - 50:mid + 50])
        # X = X[mid-11000:mid-10000] + X[mid+10000:mid+11000]
        #
        # y = y[mid-11000:mid-10000] + y[mid+10000:mid+11000]
        X = X[mid-1000:mid+1000]
        y = y[mid - 1000:mid + 1000]
    else:
        X, y = read_file(data_type)
    print(f"{len(X),len(y)}")
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
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


def evaluate(model, data_loader,device):
    model.eval()
    preds = []
    truths = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pred = model(input_ids, attention_mask=attention_mask)[0]
            pred = torch.argmax(pred,dim=1).detach().cpu().numpy().tolist()
            truth=batch['labels'].squeeze().cpu().numpy().tolist()
            if type(truth) == type(0):
                truth = [truth]
            # print(truth)
            preds.extend(pred)
            truths.extend(truth)
    return precision_recall_fscore_support(truths,preds, average=None, labels=[0,1])


def main(train_loader, valid_loader, test_loader):
    TORCH_SEED = 42
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    torch.backends.cudnn.deterministic = True


    model = XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=2)

    model.to(device)
    model.train()



    optim = AdamW(model.parameters(), lr=2e-5)
    num_training_steps = EPOCH * len(train_loader)
    print(f"Total training steps: {num_training_steps}")
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optim,
        num_warmup_steps=0.0*num_training_steps,
        num_training_steps=num_training_steps
    )

    model.train()
    best_f1,best_p, best_r = 0.0,0.0,0.0
    best_epoch = -1
    cur_step=0
    for epoch in range(EPOCH):
        epoch_start = time.time()
        print(f"epoch {epoch} starts.")
        for batch in train_loader:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)

            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optim.step()
            lr_scheduler.step()
            cur_step+=1

            if cur_step % 1000 == 0:
                # print("Start evaluating...")
                out = evaluate(model, valid_loader, device)
                out_real = evaluate(model, test_loader, device)
                # p = out[0]
                # r = out[1]
                # f1 = out[2]
                #
                # metric = f"p={p:.5f}, r={r:.5f}, f1={f1:.5f}. loss={loss}"
                #
                # print(metric)
                print(out)
                print(out_real)
                print(loss)
                # if f1 > best_f1:
                #     best_f1 = f1
                #     best_p = p
                #     best_r = r
                #     print(f"saving model for training step {cur_step}")
                #     # torch.save(model.state_dict(), os.path.join(ckpt_dir, f"checkpoint_best_xlm.pth"))

                model.train()


        print(f"-------end of Epoch {epoch}-----------")

    return best_p, best_r, best_f1


if __name__ == '__main__':

    if sys.argv[1] == 'train-full':
        train_loader = build_dataloader('train-full')
        valid_loader = build_dataloader('valid')
        test_loader = build_dataloader('test17')
        p, r, f1 = main(train_loader,valid_loader, test_loader)
        print(f"best: {p},{r},{f1}\n===============end============")