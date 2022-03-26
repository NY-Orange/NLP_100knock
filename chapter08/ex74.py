import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ex71 import SLPNet
from ex73 import NewsDataset

def make_dataloaders():
    X_train = torch.load("./data/X_train.pt")
    y_train = torch.load("./data/y_train.pt")
    X_valid = torch.load("./data/X_valid.pt")
    y_valid = torch.load("./data/y_valid.pt")
    X_test = torch.load("./data/X_test.pt")
    y_test = torch.load("./data/y_test.pt")

    # Datasetの作成
    Dataset_train = NewsDataset(X_train, y_train)
    Dataset_valid = NewsDataset(X_valid, y_valid)
    Dataset_test = NewsDataset(X_test, y_test)

    # DataLoaderの作成
    DataLoader_train = DataLoader(Dataset_train, batch_size=1, shuffle=True)
    DataLoader_valid = DataLoader(Dataset_valid, batch_size=len(Dataset_valid), shuffle=False)
    DataLoader_test = DataLoader(Dataset_test, batch_size=len(Dataset_test), shuffle=False)

    return DataLoader_train, DataLoader_valid, DataLoader_test

def calculate_acc(model, dataloader):
    model.eval()
    total_num = 0
    correct_num = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            pred = torch.argmax(outputs, dim=-1)
            total_num += len(inputs)
            correct_num += (pred==labels).sum().item()
    
    return correct_num / total_num



def main():
    word_vec_dim = 300
    label_num = 4

    DataLoader_train, DataLoader_valid, DataLoader_test = make_dataloaders()

    model = SLPNet(word_vec_dim, label_num)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    max_epoch = 10
    for epoch in range(max_epoch):
        model.train()
        loss_train = 0.0
        for inputs, labels in DataLoader_train:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_train += loss.item()
        loss_train = loss_train / len(DataLoader_train)

        model.eval()
        with torch.no_grad():
            inputs, labels = next(iter(DataLoader_valid))
            outputs = model(inputs)
            loss_valid = criterion(outputs, labels)
        
        print("epoch: {}\tloss_train: {:.3f}\tloss_valid: {:.3f}".format(epoch+1, loss_train, loss_valid))
    
    acc_train = calculate_acc(model, DataLoader_train)
    acc_test = calculate_acc(model, DataLoader_test)
    print("正解率（訓練データ）：{:.3f}".format(acc_train))
    print("正解率（評価データ）：{:.3f}".format(acc_test))



if __name__ == "__main__":
    main()