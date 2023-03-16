import torch
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import sys
import os
import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_recall_fscore_support

from transformers import BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
ckpt_dir = "checkpoint_nese"
if not os.path.exists(ckpt_dir):
    os.mkdir(ckpt_dir)
    print("Checkpoint directory established.")
# with open('sentiment_classification_result.txt','w') as f:
#     f.write("Evaluation of emotion extraction.")

DATA_DIR='/workspace/translationese/data/ours/'
EPOCH = 5
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
    Xs, ys = [], []
    with open(filename1,encoding='utf-8') as f:
        for line in f.readlines():
            if len(line) <= 128:
                Xs.append(line.strip())
                ys.append(1)
    with open(filename2,encoding='utf-8') as f:
        for line in f.readlines():
            if len(line) <= 128:
                Xs.append(line.strip())
                ys.append(0)

    return Xs, ys


def build_dataloader(data_type):

    X, y = read_file(data_type)

    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    print("Start tokenize")
    encodings = tokenizer(list(X), truncation=True, padding=True)
    print(f"{len(X),len(y)}")
    # print(X[1000:1005],y[1000:1005])
    dataset = Dataset(encodings, y)
    if data_type == 'train' or data_type == 'train-large' or data_type == 'train-filter' or data_type == 'train-full':
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        return train_loader

    elif data_type[:4] == 'test':
        valid_loader = DataLoader(dataset, batch_size=64, shuffle=False)
        return valid_loader
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
    return precision_recall_fscore_support(truths,preds, average='macro', labels=[0,1])


def main(train_loader, valid_loader):
    TORCH_SEED = 42
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    torch.backends.cudnn.deterministic = True


    model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2)

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
    best_f1 = 0.0
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
            if cur_step % 500 == 0:
                print(f"loss={loss}")
            if cur_step % 1000 == 0:
                # print("Start evaluating...")
                out = evaluate(model, valid_loader, device)
                p = out[0]
                r = out[1]
                f1 = out[2]
                metric = f"p={p:.5f}, r={r:.5f}, f1={f1:.5f}."

                update_time = (time.time() - epoch_start)
                time_loss = "time : %d min %d sec， loss = %f" % \
                            ((update_time % 3600) // 60, update_time % 60, loss)
                print(metric + time_loss)

                if f1 > best_f1:
                    best_f1 = f1;
                    best_p = p;
                    best_r = r;
                    print(f"saving model for training step {cur_step}")
                    torch.save(model.state_dict(), os.path.join(ckpt_dir, f"checkpoint_best_filter.pth"))

                model.train()

        out = evaluate(model, valid_loader, device)
        p = out[0]
        r = out[1]
        f1 = out[2]
        epoch_metric = f"p={p:.5f}, r={r:.5f}, f1={f1:.5f}."

        epoch_time = (time.time() - epoch_start)
        time_loss = "epoch time : %d min %d sec， loss = %f" % \
                    ((epoch_time % 3600) // 60, epoch_time % 60, loss)
        print(epoch_metric + time_loss)



        model.train()

    return best_p, best_r, best_f1


def predict(valid_loader):

    model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2)

    model.to(device)
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, f"checkpoint_best_full.pth")))
    out = evaluate(model, valid_loader, device)
    p = out[0]
    r = out[1]
    f1 = out[2]
    return p, r , f1

if __name__ == '__main__':
    if sys.argv[1] == 'train':
        train_loader = build_dataloader('train')
        test_loader = build_dataloader('test')
        p, r, f1 = main(train_loader,test_loader)
        print(f"best: {p},{r},{f1}\n===============end============")
    elif sys.argv[1] == 'train-large' or sys.argv[1] == 'train-filter' or sys.argv[1] == 'train-bad' or sys.argv[1] == 'train-full':
        train_loader = build_dataloader(sys.argv[1])
        test_loader = build_dataloader('test')
        p, r, f1 = main(train_loader,test_loader)
        print(f"best: {p},{r},{f1}\n===============end============")
    elif sys.argv[1] == 'predict':

        for i in range(17,21):
            test_loader = build_dataloader(f"test{i}")
            p, r , f1 = predict(test_loader)

            print(f"Translationese-encoder-3 in test{i} : {p},{r},{f1}\n=============end============")