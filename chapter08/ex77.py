import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ex71 import SLPNet
from ex73 import NewsDataset
from ex75 import calculate_loss_and_acc

def make_datasets():
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

    return Dataset_train, Dataset_valid, Dataset_test

def train_model(Dataset_train, Dataset_valid, batch_size, model, criterion, optimizer, max_epoch):
    # DataLoaderの作成（ミニバッチ化）
    DataLoader_train = DataLoader(Dataset_train, batch_size=batch_size, shuffle=True)
    DataLoader_valid = DataLoader(Dataset_valid, batch_size=len(Dataset_valid), shuffle=False)

    for epoch in range(max_epoch):
        start_time = time.time()

        model.train()
        for inputs, labels in DataLoader_train:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        loss_train, acc_train = calculate_loss_and_acc(model, criterion, DataLoader_train)
        loss_valid, acc_valid = calculate_loss_and_acc(model, criterion, DataLoader_valid)

        end_time = time.time()

        print("epoch: {}\tloss(train): {:.3f}\taccuracy(train): {:.3f}\tloss(valid): {:.3f}\taccuracy(valid): {:.3f}\ttime: {:.3f}sec".format(epoch+1, loss_train, acc_train, loss_valid, acc_valid, end_time-start_time))



def main():
    word_vec_dim = 300
    label_num = 4

    Dataset_train, Dataset_valid, _ = make_datasets()

    model = SLPNet(word_vec_dim, label_num)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    max_epoch = 1
    max_batch_size_exponent = 10

    for batch_size in [2**i for i in range(max_batch_size_exponent+1)]:
        print("バッチサイズ：{}".format(batch_size))
        train_model(Dataset_train, Dataset_valid, batch_size, model, criterion, optimizer, max_epoch)



if __name__ == "__main__":
    main()