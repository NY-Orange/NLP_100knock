import time
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from ex80 import preprocess_dataset, make_word2id_dict, text2id
from ex81 import RNN, CreateNewsDataset

def calculate_loss_and_acc(model, dataset, device=None, criterion=None):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    loss = 0.0
    total_num = 0
    correct_num = 0
    with torch.no_grad():
        for data in dataloader:
            inputs = data["inputs"].to(device)
            labels = data["labels"].to(device)

            outputs = model(inputs)

            if criterion != None:
                loss += criterion(outputs, labels).item()
            
            pred = torch.argmax(outputs, dim=-1)
            total_num += len(inputs)
            correct_num += (pred==labels).sum().item()
        
        return loss / len(dataset), correct_num / total_num

def train_model(dataset_train, dataset_valid, batch_size, model, criterion, optimizer, max_epoch, collate_fn=None, device=None):
    model.to(device)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # スケジューラの設定
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epoch, eta_min=0.00001, last_epoch=-1)

    log_train = []
    log_valid = []
    for epoch in range(max_epoch):
        start_time = time.time()

        model.train()
        for data in dataloader_train:
            optimizer.zero_grad()

            inputs = data["inputs"].to(device)
            labels = data["labels"].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        loss_train, acc_train = calculate_loss_and_acc(model, dataset_train, device, criterion=criterion)
        loss_valid, acc_valid = calculate_loss_and_acc(model, dataset_valid, device, criterion=criterion)
        log_train.append([loss_train, acc_train])
        log_valid.append([loss_valid, acc_valid])

        torch.save({"epoch":epoch, "model_state_dict":model.state_dict(), "optimizer_state_dict":optimizer.state_dict()}, "./outputs/ex82_checkpoints/checkpoint{}.pt".format(epoch+1))

        end_time = time.time()

        print("epoch: {}\tloss(train): {:.3f}\taccuracy(train): {:.3f}\tloss(valid): {:.3f}\taccuracy(valid): {:.3f}\t{:.3f}sec".format(epoch+1, loss_train, acc_train, loss_valid, acc_valid, end_time-start_time))

        # 検証データの損失が3エポック連続で低下しなかった場合は学習終了
        if epoch > 2 and log_valid[epoch-3][0] <= log_valid[epoch-2][0] <= log_valid[epoch-1][0] <= log_valid[epoch][0]:
            break

        # scheduler.step() で学習率が変化
        scheduler.step()
    
    return {"train":log_train, "valid":log_valid}

def visualize_logs(log, filename):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    for i, ylabel in enumerate(["loss", "accuracy"]):
        ax[i].plot(np.array(log["train"]).T[i], label="train")
        ax[i].plot(np.array(log["valid"]).T[i], label="valid")
        ax[i].set_xlabel("epoch")
        ax[i].set_ylabel(ylabel)
        ax[i].legend()
    
    fig.savefig("./outputs/{}".format(filename))
    plt.show()



def main():
    train, valid, test = preprocess_dataset()

    word2id_dict = make_word2id_dict(train)

    category_dict = {"b":0, "t":1, "e":2, "m":3}
    y_train = train["CATEGORY"].map(lambda x: category_dict[x]).values
    y_valid = valid["CATEGORY"].map(lambda x: category_dict[x]).values
    y_test = test["CATEGORY"].map(lambda x: category_dict[x]).values
    dataset_train = CreateNewsDataset(train["TITLE"], y_train, text2id, word2id_dict)
    dataset_valid = CreateNewsDataset(valid["TITLE"], y_valid, text2id, word2id_dict)
    dataset_test = CreateNewsDataset(test["TITLE"], y_test, text2id, word2id_dict)

    VOCAB_SIZE = len(set(word2id_dict.values())) + 1
    EMB_SIZE = 300
    PADDING_IDX = len(set(word2id_dict.values()))
    OUTPUT_SIZE = 4
    HIDDEN_SIZE = 50
    LEARNING_RATE = 0.001
    BATCH_SIZE = 1
    MAX_EPOCH = 10

    model = RNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, HIDDEN_SIZE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    log = train_model(dataset_train, dataset_valid, BATCH_SIZE, model, criterion, optimizer, MAX_EPOCH)

    visualize_logs(log, "ex82_calculation_loss_and_accuracy.png")

    _, acc_train = calculate_loss_and_acc(model, dataset_train)
    _, acc_test = calculate_loss_and_acc(model, dataset_test)
    print("正解率（学習データ）：{:.3f}".format(acc_train))
    print("正解率（評価データ）：{:.3f}".format(acc_test))



if __name__ == "__main__":
    main()