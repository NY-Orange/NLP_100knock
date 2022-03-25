import torch
import torch.nn as nn
from ex71 import SLPNet
from ex74 import make_dataloaders
from ex75 import calculate_loss_and_acc



def main():
    word_vec_dim = 300
    label_num = 4

    DataLoader_train, DataLoader_valid, _ = make_dataloaders()

    model = SLPNet(word_vec_dim, label_num)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    max_epoch = 10
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

        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, "./outputs/ex76_checkpoints/checkpoint{}.pt".format(epoch+1))

        print("epoch: {}\tloss(train): {:.3f}\taccuracy(train): {:.3f}\tloss(valid): {:.3f}\taccuracy(valid): {:.3f}".format(epoch+1, loss_train, acc_train, loss_valid, acc_valid))



if __name__ == "__main__":
    main()