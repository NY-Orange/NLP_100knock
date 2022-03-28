import optuna
import torch
import torch.nn as nn
from torch.nn import functional as F
from ex80 import preprocess_dataset, make_word2id_dict, text2id
from ex81 import CreateNewsDataset
from ex82 import calculate_loss_and_acc, train_model, visualize_logs
from ex83 import Padsequence
from ex84 import make_emb_weights

train, valid, test = preprocess_dataset()
word2id_dict = make_word2id_dict(train)

VOCAB_SIZE = len(set(word2id_dict.values())) + 1
EMB_SIZE = 300
PADDING_IDX = len(set(word2id_dict.values()))
OUTPUT_SIZE = 4
CONV_PARAMS = [[2, 0], [3, 1], [4, 2]]
MAX_EPOCH = 30

weights  = make_emb_weights(word2id_dict, VOCAB_SIZE, EMB_SIZE)

category_dict = {"b":0, "t":1, "e":2, "m":3}
y_train = train["CATEGORY"].map(lambda x: category_dict[x]).values
y_valid = valid["CATEGORY"].map(lambda x: category_dict[x]).values
y_test = test["CATEGORY"].map(lambda x: category_dict[x]).values
dataset_train = CreateNewsDataset(train["TITLE"], y_train, text2id, word2id_dict)
dataset_valid = CreateNewsDataset(valid["TITLE"], y_valid, text2id, word2id_dict)
dataset_test = CreateNewsDataset(test["TITLE"], y_test, text2id, word2id_dict)

class textCNN(nn.Module):
    def __init__(self, vocab_size, emb_size, padding_idx, output_size, out_channels, conv_params, drop_rate, emb_weights=None):
        super().__init__()
        if emb_weights != None:
            self.emb = nn.Embedding.from_pretrained(emb_weights, padding_idx=padding_idx)
        else:
            self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        self.convs = nn.ModuleList([nn.Conv2d(1, out_channels, (kernel_height, emb_size), padding=(padding, 0)) for kernel_height, padding in conv_params])
        self.drop = nn.Dropout(drop_rate)
        self.fc = nn.Linear(len(conv_params) * out_channels, output_size)
    
    def forward(self, x):
        # x.size() = (batch_size, seq_len)
        emb = self.emb(x).unsqueeze(1)
        # emb.size() = (batch?size, 1, seq_len, emb_size)
        conv = [F.relu(conv(emb)).squeeze(3) for i, conv in enumerate(self.convs)]
        # conv[i].size() = (batch_size, out_channels, seq_len + padding*2 - kernel_height + 1)
        max_pool = [F.max_pool1d(i, i.size(2)) for i in conv]
        # max_pool[i].size() = (batch_size, out_channels, 1) -> seq_len方向に最大値を取得
        max_pool_cat = torch.cat(max_pool, 1)
        # max_pool_cat.size() = (batch_size, len(conv_params)*out_channels, 1) -> フィルター別の結果を結合
        out = self.fc(self.drop(max_pool_cat.squeeze(2)))
        # out.size() = (batch_size, output_size

        return out

def objective(trial):
    out_channels = int(trial.suggest_discrete_uniform("out_channels", 50, 200, 50))
    drop_rate = trial.suggest_discrete_uniform("drop_rate", 0.0, 0.5, 0.1)
    learning_rate = trial.suggest_loguniform("learning_rate", 0.0005, 0.05)
    momentum = trial.suggest_discrete_uniform("momentum", 0.5, 0.9, 0.1)
    batch_size = int(trial.suggest_discrete_uniform("batch_size", 16, 128, 16))

    model = textCNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, out_channels, CONV_PARAMS, drop_rate, emb_weights=weights)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    device = torch.device("cuda")

    log = train_model(dataset_train, dataset_valid, batch_size, model, criterion, optimizer, MAX_EPOCH, collate_fn=Padsequence(PADDING_IDX), device=device)

    loss_valid, _ = calculate_loss_and_acc(model, dataset_valid, device, criterion=criterion)

    return loss_valid

def main():
    study = optuna.create_study()
    study.optimize(objective, timeout=600)

    print("Best trial:")
    trial = study.best_trial
    print("  Value: {:.3f}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}:{}".format(key, value))
    
    OUT_CHANNELS = int(trial.params["out_channels"])
    DROP_RATE = trial.params["drop_rate"]
    LEARNING_RATE = trial.params["learning_rate"]
    BATCH_SIZE = int(trial.params["batch_size"])
    MOMENTUM = trial.params["momentum"]

    model = textCNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, OUT_CHANNELS, CONV_PARAMS, DROP_RATE, emb_weights=weights)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    device = torch.device("cuda")

    log = train_model(dataset_train, dataset_valid, BATCH_SIZE, model, criterion, optimizer, MAX_EPOCH, collate_fn=Padsequence(PADDING_IDX), device=device)

    visualize_logs(log, "ex88_calculation_loss_and_accuracy.png")

    _, acc_train = calculate_loss_and_acc(model, dataset_train, device)
    _, acc_test = calculate_loss_and_acc(model, dataset_test, device)
    print("正解率（学習データ）：{:.3f}".format(acc_train))
    print("正解率（評価データ）：{:.3f}".format(acc_test))



if __name__ == "__main__":
    main()