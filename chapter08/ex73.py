import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from ex71 import SLPNet

class NewsDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    # len(dataset) で返す値を指定
    def __len__(self):
        return len(self.y)
    
    # dataset[idx] で返す値を指定
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

def main():
    word_vec_dim = 300
    label_num = 4

    X_train = torch.load("./data/X_train.pt")
    y_train = torch.load("./data/y_train.pt")
    X_valid = torch.load("./data/X_valid.pt")
    y_valid = torch.load("./data/y_valid.pt")

    # Datasetの作成
    Dataset_train = NewsDataset(X_train, y_train)
    Dataset_valid = NewsDataset(X_valid, y_valid)

    # DataLoaderの作成
    DataLoader_train = DataLoader(Dataset_train, batch_size=1, shuffle=True)
    DataLoader_valid = DataLoader(Dataset_valid, batch_size=len(Dataset_valid), shuffle=False)

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
    
    print("学習された重み行列\n{}".format(model.fc.weight))



if __name__ == "__main__":
    main()