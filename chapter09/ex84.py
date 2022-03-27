import numpy as np
import torch
import torch.nn as nn
from gensim.models import KeyedVectors
from ex80 import preprocess_dataset, make_word2id_dict, text2id
from ex81 import CreateNewsDataset
from ex82 import calculate_loss_and_acc, train_model, visualize_logs
from ex83 import Padsequence

class RNN(nn.Module):
    def __init__(self, vocab_size, emb_size, padding_idx, output_size, hidden_size, num_layers, emb_weights=None, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 単方向：1, 双方向：2
        self.num_directions = bidirectional + 1
        if emb_weights != None:
            self.emb = nn.Embedding.from_pretrained(emb_weights, padding_idx=padding_idx)
        else:
            self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        self.rnn = nn.RNN(emb_size, hidden_size, num_layers, nonlinearity="tanh", bidirectional=bidirectional, batch_first=True)
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)
    
    def forward(self, x):
        device = x.device
        self.batch_size = x.size()[0]
        hidden = self.init_hidden(device)
        emb = self.emb(x)
        out, hidden = self.rnn(emb, hidden)
        out = self.fc(out[:, -1, :])
        
        return out
    
    def init_hidden(self, device):
        hidden = torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_size, device=device)
        return hidden

def make_emb_weights(word2id_dict, VOCAB_SIZE, EMB_SIZE):
    word_vec_model = KeyedVectors.load_word2vec_format("./datasets/GoogleNews-vectors-negative300.bin.gz", binary=True)
    weights  = np.zeros((VOCAB_SIZE, EMB_SIZE))
    words_in_pretrained = 0
    for i, word in enumerate(word2id_dict.keys()):
        try:
            weights[i] = word_vec_model[word]
            words_in_pretrained += 1
        except KeyError:
            weights[i] = np.random.normal(scale=0.4, size=(EMB_SIZE, ))
    weights = torch.from_numpy(weights.astype(np.float32))

    print("学習済みベクトル利用単語数：{} / {}".format(words_in_pretrained, VOCAB_SIZE))
    print(weights.size())

    return weights



def main():
    train, valid, test = preprocess_dataset()

    word2id_dict = make_word2id_dict(train)


    VOCAB_SIZE = len(set(word2id_dict.values())) + 1
    EMB_SIZE = 300

    weights = make_emb_weights(word2id_dict, VOCAB_SIZE, EMB_SIZE)

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
    NUM_LAYERS = 1
    LEARNING_RATE = 0.05
    BATCH_SIZE = 32
    MAX_EPOCH = 10

    model = RNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, emb_weights=weights)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    device = torch.device("cuda")

    # RNN 側で hidden を gpu に送る処理をしないとエラーが出る
    log = train_model(dataset_train, dataset_valid, BATCH_SIZE, model, criterion, optimizer, MAX_EPOCH, collate_fn=Padsequence(PADDING_IDX), device=device)

    visualize_logs(log, "ex84_calculation_loss_and_accuracy.png")

    _, acc_train = calculate_loss_and_acc(model, dataset_train, device)
    _, acc_test = calculate_loss_and_acc(model, dataset_test, device)
    print("正解率（学習データ）：{:.3f}".format(acc_train))
    print("正解率（評価データ）：{:.3f}".format(acc_test))



if __name__ == "__main__":
    main()