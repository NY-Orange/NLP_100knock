import torch
import torch.nn as nn
from ex80 import preprocess_dataset, make_word2id_dict, text2id
from ex81 import CreateNewsDataset
from ex82 import calculate_loss_and_acc, train_model, visualize_logs
from ex83 import Padsequence
from ex84 import RNN, make_emb_weights



def main():
    train, valid, test = preprocess_dataset()

    word2id_dict = make_word2id_dict(train)

    VOCAB_SIZE = len(set(word2id_dict.values())) + 1
    EMB_SIZE = 300
    weights  = make_emb_weights(word2id_dict, VOCAB_SIZE, EMB_SIZE)

    category_dict = {"b":0, "t":1, "e":2, "m":3}
    y_train = train["CATEGORY"].map(lambda x: category_dict[x]).values
    y_valid = valid["CATEGORY"].map(lambda x: category_dict[x]).values
    y_test = test["CATEGORY"].map(lambda x: category_dict[x]).values
    dataset_train = CreateNewsDataset(train["TITLE"], y_train, text2id, word2id_dict)
    dataset_valid = CreateNewsDataset(valid["TITLE"], y_valid, text2id, word2id_dict)
    dataset_test = CreateNewsDataset(test["TITLE"], y_test, text2id, word2id_dict)

    PADDING_IDX = len(set(word2id_dict.values()))
    OUTPUT_SIZE = 4
    HIDDEN_SIZE = 50
    NUM_LAYERS = 2
    LEARNING_RATE = 0.05
    BATCH_SIZE = 32
    MAX_EPOCH = 10

    model = RNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, emb_weights=weights, bidirectional=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    device = torch.device("cuda")

    # RNN 側で hidden を GPU に送る処理をしないとエラーが出る
    log = train_model(dataset_train, dataset_valid, BATCH_SIZE, model, criterion, optimizer, MAX_EPOCH, collate_fn=Padsequence(PADDING_IDX), device=device)

    visualize_logs(log, "ex85_calculation_loss_and_accuracy.png")

    _, acc_train = calculate_loss_and_acc(model, dataset_train, device)
    _, acc_test = calculate_loss_and_acc(model, dataset_test, device)
    print("正解率（学習データ）：{:.3f}".format(acc_train))
    print("正解率（評価データ）：{:.3f}".format(acc_test))



if __name__ == "__main__":
    main()