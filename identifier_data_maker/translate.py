from transformers import MarianTokenizer, MarianMTModel
from torch.utils.data import DataLoader
import os
import time
import sys
import torch
from tqdm import tqdm

# src = "en"  # source language
# trg = "jap"  # target language
FILE = sys.argv[1]
OUT_DIR = sys.argv[2]
src=sys.argv[3]
tgt=sys.argv[4]

model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]).to(device) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


def read_file(filename):
    res = []

    with open(filename,mode='r',encoding='utf-8') as f:
        for line in f.readlines():
            # if len(line)<=128:
                res.append(line.strip())

    return res


def build_dataloader(filename):

    X = read_file(filename)
    print(len(X))

    # X = X[int(start):int(end)]
    print(len(X))
    print("Start tokenize")
    encodings = tokenizer(list(X), padding=True, truncation=True)
    dataset = Dataset(encodings)
    data_loader = DataLoader(dataset=dataset,batch_size=64,shuffle=False)
    return data_loader

def main(data_loader,out_dir):
    TORCH_SEED = 42
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    torch.backends.cudnn.deterministic = True
    model.to(device)
    res = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            generated_ids = model.generate(**batch)
            # print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
            res.extend(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))

    with open(out_dir,mode='w',encoding='utf-8') as f:
        for line in res:
            f.write(line+'\n')



if __name__ == '__main__':

    dataloader = build_dataloader(filename=FILE)
    main(dataloader,out_dir=OUT_DIR)