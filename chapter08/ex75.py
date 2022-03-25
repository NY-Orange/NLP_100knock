import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from ex71 import SLPNet
from ex74 import make_dataloaders

def calculate_loss_and_acc(model, criterion, dataloader):
    model.eval()
    loss = 0.0
    total_num = 0
    correct_num = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            loss += criterion(outputs, labels).item()
            pred = torch.argmax(outputs, dim=-1)
            total_num += len(inputs)
            correct_num += (pred==labels).sum().item()
    
    return loss / len(dataloader), correct_num / total_num



def main():
    word_vec_dim = 300
    label_num = 4

    DataLoader_train, DataLoader_valid, _ = make_dataloaders()

    model = SLPNet(word_vec_dim, label_num)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    max_epoch = 30
    log_train = []
    log_valid = []
    for epoch in range(max_epoch):
        model.train()
        for inputs, labels in DataLoader_train:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        loss_train, acc_train = calculate_loss_and_acc(model, criterion, DataLoader_train)
        loss_valid, acc_valid = calculate_loss_and_acc(model, criterion, DataLoader_valid)
        log_train.append([loss_train, acc_train])
        log_valid.append([loss_valid, acc_valid])

        print("epoch: {}\tloss(train): {:.3f}\taccuracy(train): {:.3f}\tloss(valid): {:.3f}\taccuracy(valid): {:.3f}".format(epoch+1, loss_train, acc_train, loss_valid, acc_valid))
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    for i, ylabel in enumerate(["loss", "accuracy"]):
        ax[i].plot(np.array(log_train).T[i], label="train")
        ax[i].plot(np.array(log_valid).T[i], label="valid")
        ax[i].set_xlabel("epoch")
        ax[i].set_ylabel(ylabel)
        ax[i].legend()
    
    fig.savefig("./outputs/ex75_calculation_loss_and_accuracy.png")



if __name__ == "__main__":
    main()