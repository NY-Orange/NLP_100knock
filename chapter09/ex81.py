import torch
import torch.nn as nn
from torch.utils.data import Dataset
from ex80 import preprocess_dataset, make_word2id_dict, text2id

class RNN(nn.Module):
    def __init__(self, vocab_size, emb_size, padding_idx, output_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        self.rnn = nn.RNN(emb_size, hidden_size, nonlinearity="tanh", batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x と hidden のdevice を同じにしないとエラーが出る
        device = x.device
        self.batch_size = x.size()[0]
        hidden = self.init_hidden(device)
        emb = self.emb(x)
        # emb.size() = (batch_size, seq_len, emb_size)
        out, hidden = self.rnn(emb, hidden)
        # out.size() = (batch_size, seq_len, hidden_size)
        out = self.fc(out[:, -1, :])
        # out.size() = (batch_size, output_size)

        return out
    
    def init_hidden(self, device):
        hidden = torch.zeros(1, self.batch_size, self.hidden_size, device=device)
        return hidden

class CreateNewsDataset(Dataset):
    def __init__(self, X, y, text2id, word2id_dict):
        self.X = X
        self.y = y
        self.text2id = text2id
        self.word2id_dict = word2id_dict
    
    # len(dataset) で返す値を指定
    def __len__(self):
        return len(self.y)
    
    # dataset[idx] で返す値を指定
    def __getitem__(self, index):
        # self.X[index] だと KeyError
        text = self.X.iloc[index]
        inputs = self.text2id(text, self.word2id_dict)
        return_dic = {
            "inputs":torch.tensor(inputs, dtype=torch.int64),
            "labels":torch.tensor(self.y[index], dtype=torch.int64)
        }

        return return_dic



def main():
    train, _, _ = preprocess_dataset()

    word2id_dict = make_word2id_dict(train)
    
    category_dict = {"b":0, "t":1, "e":2, "m":3}
    y_train = train["CATEGORY"].map(lambda x: category_dict[x]).values
    dataset_train = CreateNewsDataset(train["TITLE"], y_train, text2id, word2id_dict)

    print("len(Dataset) の出力：{}".format(len(dataset_train)))
    print("Dataset[index] の出力：")
 
    # 辞書のID数 + パディングID
    VOCAB_SIZE = len(set(word2id_dict.values())) + 1
    EMB_SIZE = 300
    PADDING_IDX = len(word2id_dict.values())
    OUTPUT_SIZE = 4
    HIDDEN_SIZE = 50

    model = RNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, HIDDEN_SIZE)

    for i in range(10):
        X = dataset_train[i]["inputs"]
        print(torch.softmax(model(X.unsqueeze(0)), dim=-1))



if __name__ == "__main__":
    main()