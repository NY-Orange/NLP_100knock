import time
import pandas as pd
import torch
import torch.nn as nn
from torch import cuda
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from ex80 import preprocess_dataset, make_word2id_dict
from ex82 import visualize_logs

train, valid, test = preprocess_dataset()
word2id_dict = make_word2id_dict(train)

class CreateNewsDataset(Dataset):
    def __init__(self, X, y, tokenizer, max_len):
        self.X = X
        self.y = y
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    # len(dataset) で返す値を指定
    def __len__(self):
        return len(self.y)
    
    # dataset[idx] で返す値を指定
    def __getitem__(self, index):
        # self.X[index] だと KeyError
        text = self.X.iloc[index]
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        return_dic = {
            "ids":torch.LongTensor(ids),
            "mask":torch.LongTensor(mask),
            "labels":torch.Tensor(self.y[index])
        }

        return return_dic

class BertClassifier(nn.Module):
    def __init__(self, drop_rate, output_size):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.drop = nn.Dropout(drop_rate)
        self.fc = nn.Linear(768, output_size)
    
    def forward(self, ids, mask):
        _, out = self.bert(ids, attention_mask=mask, return_dict=False)
        out = self.drop(out)
        out = self.fc(out)
        return out

def calculate_loss_and_acc(model, criterion, dataloader, device):
    model.eval()
    loss = 0.0
    total_num = 0
    correct_num = 0
    with torch.no_grad():
        for data in dataloader:
            ids = data["ids"].to(device)
            mask = data["mask"].to(device)
            labels = data["labels"].to(device)

            outputs = model(ids, mask)
            loss += criterion(outputs, labels).item()

            pred = torch.argmax(outputs, dim=-1).cpu().numpy()
            labels = torch.argmax(labels, dim=-1).cpu().numpy()
            total_num += len(labels)
            correct_num += (pred==labels).sum().item()
    
    return loss / len(dataloader), correct_num / total_num

def train_model(dataset_train, dataset_valid, batch_size, model, criterion, optimizer, max_epoch, device=None):
    model.to(device)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=len(dataset_valid), shuffle=False)

    log_train = []
    log_valid = []
    for epoch in range(max_epoch):
        start_time = time.time()

        model.train()
        for data in dataloader_train:
            ids = data["ids"].to(device)
            mask = data["mask"].to(device)
            labels = data["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(ids, mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        loss_train, acc_train = calculate_loss_and_acc(model, criterion, dataloader_train, device)
        loss_valid, acc_valid = calculate_loss_and_acc(model, criterion, dataloader_valid, device)
        log_train.append([loss_train, acc_train])
        log_valid.append([loss_valid, acc_valid])

        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, "./outputs/ex89_checkpoints/checkpoint{}.pt".format(epoch+1))

        end_time = time.time()

        print(f"epoch: {epoch+1}\tloss(train): {loss_train:.3f}\taccuracy(train): {acc_train:.3f}\tloss(valid): {loss_valid:.3f}\taccuracy(valid): {acc_valid:.3f}\t{end_time-start_time:.3f}sec")

    return {"train":log_train, "valid":log_valid}

def calculate_acc(model, dataset, device):
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)

    model.eval()
    total_num = 0
    correct_num = 0
    with torch.no_grad():
        for data in dataloader:
            ids = data["ids"].to(device)
            mask = data["mask"].to(device)
            labels = data["labels"].to(device)

            outputs = model.forward(ids, mask)
            pred = torch.argmax(outputs, dim=-1).cpu().numpy()
            labels = torch.argmax(labels, dim=-1).cpu().numpy()
            total_num += len(labels)
            correct_num += (pred==labels).sum().item()
    
    return correct_num / total_num



def main():
    y_train = pd.get_dummies(train, columns=["CATEGORY"])[["CATEGORY_b", "CATEGORY_e", "CATEGORY_t", "CATEGORY_m"]].values
    y_valid = pd.get_dummies(valid, columns=["CATEGORY"])[["CATEGORY_b", "CATEGORY_e", "CATEGORY_t", "CATEGORY_m"]].values
    y_test = pd.get_dummies(test, columns=["CATEGORY"])[["CATEGORY_b", "CATEGORY_e", "CATEGORY_t", "CATEGORY_m"]].values

    max_len = 20
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset_train = CreateNewsDataset(train["TITLE"], y_train, tokenizer, max_len)
    dataset_valid = CreateNewsDataset(valid["TITLE"], y_valid, tokenizer, max_len)
    dataset_test = CreateNewsDataset(test["TITLE"], y_test, tokenizer, max_len)

    for var in dataset_train[0]:
        print("{}: {}".format(var, dataset_train[0][var]))

    DROP_RATE = 0.4
    OUTPUT_SIZE = 4
    BATCH_SIZE = 32
    MAX_EPOCH = 4
    LEARNING_RATE = 0.00002
    
    model = BertClassifier(DROP_RATE, OUTPUT_SIZE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)
    device = "cuda" if cuda.is_available() else "cpu"

    log = train_model(dataset_train, dataset_valid, BATCH_SIZE, model, criterion, optimizer, MAX_EPOCH, device=device)
    
    visualize_logs(log, "ex89_calculation_loss_and_accuracy.png")

    print("正解率（評価データ）：{:.3f}".format(calculate_acc(model, dataset_test, device)))



if __name__ == "__main__":
    main()