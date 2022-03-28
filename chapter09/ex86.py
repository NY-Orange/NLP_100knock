import torch
import torch.nn as nn
from torch.nn import functional as F
from ex80 import preprocess_dataset, make_word2id_dict, text2id
from ex81 import CreateNewsDataset
from ex84 import make_emb_weights

class CNN(nn.Module):
    def __init__(self, vocab_size, emb_size, padding_idx, output_size, out_channels, kernel_heights, stride, padding, emb_weights=None):
        super().__init__()
        if emb_weights != None:
            self.emb = nn.Embedding.from_pretrained(emb_weights, padding_idx=padding_idx)
        else:
            self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        self.conv = nn.Conv2d(1, out_channels, (kernel_heights, emb_size), stride, (padding, 0))
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(out_channels, output_size)
    
    def forward(self, x):
        # x.size() = (batch_size, seq_len)
        emb = self.emb(x).unsqueeze(1)
        # emb.size() = (batch_size, 1, seq_len, emb_size)
        conv = self.conv(emb)
        # conv.size() = (batch_size, out_channels, seq_len, 1)
        act = F.relu(conv.squeeze(3))
        # act.size() = (batch_size, out_channels, seq_len)
        max_pool = F.max_pool1d(act, act.size()[2])
        # max_pool.size() = (batch_size, out_channels, 1) -> seq_len方向に最大値を取得
        out = self.fc(self.drop(max_pool.squeeze(2)))
        # out.size() = (batch_size, output_size)
        
        return out


def main():
    train, _, _ = preprocess_dataset()

    word2id_dict = make_word2id_dict(train)

    VOCAB_SIZE = len(set(word2id_dict.values())) + 1
    EMB_SIZE = 300
    weights  = make_emb_weights(word2id_dict, VOCAB_SIZE, EMB_SIZE)

    category_dict = {"b":0, "t":1, "e":2, "m":3}
    y_train = train["CATEGORY"].map(lambda x: category_dict[x]).values
    dataset_train = CreateNewsDataset(train["TITLE"], y_train, text2id, word2id_dict)

    PADDING_IDX = len(set(word2id_dict.values()))
    OUTPUT_SIZE = 4
    OUT_CHANNELS = 100
    KERNEL_HEIGHTS = 3
    STRIDE = 1
    PADDING = 1

    model = CNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, OUT_CHANNELS, KERNEL_HEIGHTS, STRIDE, PADDING, emb_weights=weights)

    for i in range(10):
        x = dataset_train[i]["inputs"]
        print(torch.softmax(model(x.unsqueeze(0)), dim=-1))



if __name__ == "__main__":
    main()