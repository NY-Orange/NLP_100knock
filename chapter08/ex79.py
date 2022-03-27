import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import optuna
from ex77 import make_datasets
from ex78 import calculate_loss_and_acc_with_gpu

class MLPNet(nn.Module):
    def __init__(self, input_size, mid_size, output_size, mid_layers, dropout_rate):
        super().__init__()
        self.mid_layers = mid_layers
        # bias=Trueが初期値
        self.fc = nn.Linear(input_size, mid_size)
        self.fc_mid = nn.Linear(mid_size, mid_size)
        self.fc_out = nn.Linear(mid_size, output_size)
        self.bn = nn.BatchNorm1d(mid_size)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        h = F.relu(self.fc(x))
        h = self.dropout(h)
        for _ in range(self.mid_layers):
            h = F.relu(self.bn(self.fc_mid(h)))
            h = self.dropout(h)
        y = self.fc_out(h)

        return y

def objective(trial):
    word_vec_dim = 300
    label_num = 4
    mid_size = 200
    mid_layers = 1
    dropout_rate = 0.2
    batch_size = 64

    Dataset_train, Dataset_valid, Dataset_test = make_datasets()

    # DataLoaderの作成（ミニバッチ化）
    DataLoader_train = DataLoader(Dataset_train, batch_size=batch_size, shuffle=True)
    DataLoader_valid = DataLoader(Dataset_valid, batch_size=len(Dataset_valid), shuffle=False)
    DataLoader_test = DataLoader(Dataset_test, batch_size=len(Dataset_test), shuffle=False)

    device = torch.device("cuda")
    criterion = nn.CrossEntropyLoss()

    optimizer = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    # q=0.00001 ごとに[0.00001, 0.1]の範囲で最適化
    learning_rate = trial.suggest_discrete_uniform("learning_rate", 0.00001, 0.1, q=0.00001)

    model = MLPNet(word_vec_dim, mid_size, label_num, mid_layers, dropout_rate).to(device)
    optimizer = getattr(torch.optim, optimizer)(model.parameters(), learning_rate)

    max_epoch = 300
    model.train()
    for epoch in range(max_epoch):
        for inputs, labels in DataLoader_train:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    loss_train, acc_train = calculate_loss_and_acc_with_gpu(model, criterion, DataLoader_train, device)
    loss_valid, acc_valid = calculate_loss_and_acc_with_gpu(model, criterion, DataLoader_valid, device)

    print("loss(train): {:.3f}\taccuracy(train): {:.3f}\tloss(valid): {:.3f}\taccuracy(valid): {:.3f}".format(loss_train, acc_train, loss_valid, acc_valid))

    total_num = 0
    correct_num = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in DataLoader_test:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            pred = torch.argmax(outputs, dim=-1)
            total_num += len(inputs)
            correct_num += (pred==labels).sum().item()
    
    print(" --- accuracy(test) --- : {:.3f}".format(correct_num/total_num))

    return acc_valid



def main():
    # 重みの初期値などを固定 manual_url = https://pytorch.org/docs/stable/generated/torch.manual_seed.html
    torch.manual_seed(0)

    # Optunaで乱数シードを固定 sampler=optuna.samplers.TPESampler(seed=0)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, timeout=1800)

    trial = study.best_trial
    print("best _accuracy(valid) : {:.3f}".format(trial.value))
    print("best_parameters : {}".format(trial.params))

    word_vec_dim = 300
    label_num = 4
    mid_size = 200
    mid_layers = 1
    dropout_rate = 0.2
    batch_size = 64

    Dataset_train, Dataset_valid, Dataset_test = make_datasets()

    # DataLoaderの作成（ミニバッチ化）
    DataLoader_train = DataLoader(Dataset_train, batch_size=batch_size, shuffle=True)
    DataLoader_valid = DataLoader(Dataset_valid, batch_size=len(Dataset_valid), shuffle=False)
    DataLoader_test = DataLoader(Dataset_test, batch_size=len(Dataset_test), shuffle=False)

    device = torch.device("cuda")
    criterion = nn.CrossEntropyLoss()

    model = MLPNet(word_vec_dim, mid_size, label_num, mid_layers, dropout_rate).to(device)
    optimizer = getattr(torch.optim, trial.params["optimizer"])(model.parameters(), trial.params["learning_rate"])

    max_epoch = 300
    model.train()
    for epoch in range(max_epoch):
        for inputs, labels in DataLoader_train:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    loss_train, acc_train = calculate_loss_and_acc_with_gpu(model, criterion, DataLoader_train, device)
    loss_valid, acc_valid = calculate_loss_and_acc_with_gpu(model, criterion, DataLoader_valid, device)

    print("loss(train): {:.3f}\taccuracy(train): {:.3f}\tloss(valid): {:.3f}\taccuracy(valid): {:.3f}".format(loss_train, acc_train, loss_valid, acc_valid))

    total_num = 0
    correct_num = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in DataLoader_test:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            pred = torch.argmax(outputs, dim=-1)
            total_num += len(inputs)
            correct_num += (pred==labels).sum().item()
    
    print(" --- accuracy(test) --- : {:.3f}".format(correct_num/total_num))

if __name__ == "__main__":
    main()