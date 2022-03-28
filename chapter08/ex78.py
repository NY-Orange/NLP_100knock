import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ex71 import SLPNet
from ex77 import make_datasets

def calculate_loss_and_acc_with_gpu(model, criterion, dataloader, device):
    loss = 0.0
    total_num = 0
    correct_num = 0

    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss += criterion(outputs, labels).item()
            pred = torch.argmax(outputs, dim=-1)
            total_num += len(inputs)
            correct_num += (pred==labels).sum().item()
    
    return loss / len(dataloader), correct_num / total_num

def train_model_with_gpu(Dataset_train, Dataset_valid, batch_size, model, criterion, optimizer, max_epoch, device=None):
    model.to(device)

    # DataLoaderの作成（ミニバッチ化）
    DataLoader_train = DataLoader(Dataset_train, batch_size=batch_size, shuffle=True)
    DataLoader_valid = DataLoader(Dataset_valid, batch_size=len(Dataset_valid), shuffle=False)

    for epoch in range(max_epoch):
        start_time = time.time()

        model.train()
        for inputs, labels in DataLoader_train:
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        loss_train, acc_train = calculate_loss_and_acc_with_gpu(model, criterion, DataLoader_train, device)
        loss_valid, acc_valid = calculate_loss_and_acc_with_gpu(model, criterion, DataLoader_valid, device)

        end_time = time.time()

        print("epoch: {}\tloss(train): {:.3f}\taccuracy(train): {:.3f}\tloss(valid): {:.3f}\taccuracy(valid): {:.3f}\ttime: {:.3f}sec".format(epoch+1, loss_train, acc_train, loss_valid, acc_valid, end_time-start_time))



def main():
    word_vec_dim = 300
    label_num = 4

    Dataset_train, Dataset_valid, _ = make_datasets()

    model = SLPNet(word_vec_dim, label_num)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    device = torch.device("cuda")

    max_epoch = 1
    max_batch_size_exponent = 10

    for batch_size in [2**i for i in range(max_batch_size_exponent+1)]:
        print("バッチサイズ：{}".format(batch_size))
        train_model_with_gpu(Dataset_train, Dataset_valid, batch_size, model, criterion, optimizer, max_epoch, device=device)



if __name__ == "__main__":
    main()