import torch
import torch.nn as nn
from ex80 import preprocess_dataset, make_word2id_dict, text2id
from ex81 import RNN, CreateNewsDataset
from ex82 import calculate_loss_and_acc, train_model, visualize_logs

class Padsequence():
    def __init__(self, padding_idx):
        self.padding_idx = padding_idx
    
    def __call__(self, batch):
        sorted_batch = sorted(batch, key=lambda x: x["inputs"].shape[0], reverse=True)
        sequences = [x["inputs"] for x in sorted_batch]
        sequences_padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=self.padding_idx)
        labels = torch.LongTensor([x["labels"] for x in sorted_batch])

        return {"inputs": sequences_padded, "labels": labels}



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
    LEARNING_RATE = 0.05
    BATCH_SIZE = 16
    MAX_EPOCH = 10

    model = RNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, HIDDEN_SIZE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    device = torch.device("cuda")

    # RNN 側で hidden を gpu に送る処理をしないとエラーが出る
    log = train_model(dataset_train, dataset_valid, BATCH_SIZE, model, criterion, optimizer, MAX_EPOCH, collate_fn=Padsequence(PADDING_IDX), device=device)

    visualize_logs(log, "ex83_calculation_loss_and_accuracy.png")

    _, acc_train = calculate_loss_and_acc(model, dataset_train, device)
    _, acc_test = calculate_loss_and_acc(model, dataset_test, device)
    print("正解率（学習データ）：{:.3f}".format(acc_train))
    print("正解率（評価データ）：{:.3f}".format(acc_test))



if __name__ == "__main__":
    main()