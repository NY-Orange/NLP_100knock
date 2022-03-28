import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torchtext.data import Example, Field, Dataset, BucketIterator
from ex80 import preprocess_dataset, make_word2id_dict, text2id

class RNN(pl.LightningModule):
    def __init__(self, vocab_size, emb_size, padding_idx, output_size, hidden_size, num_layers, dropout, bidirectional=False):
        super(RNN, self).__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_size, padding_idx=padding_idx)
        # self.rnn = nn.RNN(emb_size, hidden_size, num_layers, nonlinearity="tanh", bidirectional=bidirectional, batch_first=True)
        self.lstm = nn.LSTM(input_size=emb_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
        # self.fc = nn.Linear(hidden_size * self.num_directions, output_size)
        self.fc = nn.Linear(in_features=hidden_size * (2 if bidirectional==True else 1), out_features=output_size)
    
    def forward(self, x):
        o, (h, c) = self.lstm(self.embed(x))
        return self.fc(o[-1])
    
    def training_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = self.lossfun(y, t)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = self.lossfun(y, t)
        self.log("val_loss", loss)
    
    def test_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        y = torch.argmax(y, dim=1)
        accuracy = torch.sum(y==t).item() / (len(y) * 1.0)
        self.log("test_acc", accuracy)
    
    def lossfun(self, y, t):
        return torch.nn.functional.cross_entropy(y, t)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)

def CreateNewsDataset(source_data, word2id_dict, fields):
    category_dict = {"b":0, "t":1, "e":2, "m":3}
    examples = list()

    for title, label in zip(source_data["TITLE"], source_data["CATEGORY"]):
        word_list = text2id(title, word2id_dict)
        label_id = category_dict[label]
        examples.append(Example.fromlist([word_list, label_id], fields))
    return Dataset(examples, fields)



def main():
    train, valid, test = preprocess_dataset()

    word2id_dict = make_word2id_dict(train)

    text_field = Field(sequential=True, use_vocab=True)
    label_field = Field(sequential=False, use_vocab=False, is_target=True)
    fields = [("x", text_field), ("t", label_field)]

    dataset_train = CreateNewsDataset(train, word2id_dict, fields)
    dataset_valid = CreateNewsDataset(valid, word2id_dict, fields)
    dataset_test = CreateNewsDataset(test, word2id_dict, fields)

    text_field.build_vocab(dataset_train, min_freq=1)
    
    batch_size = 64
    dataloader_train = BucketIterator(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_valid = BucketIterator(dataset_valid, batch_size=batch_size, shuffle=False)
    dataloader_test = BucketIterator(dataset_test, batch_size=batch_size, shuffle=False)

    VOCAB_SIZE = len(text_field.vocab)
    EMB_SIZE = 300
    PADDING_IDX = 1
    OUTPUT_SIZE = 4
    HIDDEN_SIZE = 300
    NUM_LAYERS = 1
    MAX_EPOCH = 10
    DROPOUT = 0

    # def RNN(self, vocab_size, emb_size, padding_idx, output_size, hidden_size, num_layers, dropout, bidirectional=False):
    model = RNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, bidirectional=False)

    checkpoint = pl.callbacks.ModelCheckpoint(
        # 検証用データにおける損失が最も小さいモデルを保存する
        monitor = "val_loss", mode="min", save_top_k=1,
        # モデルファイル（重みのみ）を"./ex88_LSTM_checkpoint"に保存
        save_weights_only=True, dirpath="./outputs/ex88_LSTM_checkpoint"
    )

    trainer = pl.Trainer(gpus=1, max_epochs=MAX_EPOCH, callbacks=[checkpoint])
    trainer.fit(model, dataloader_train, dataloader_valid)

    print("ベストモデル：{}".format(checkpoint.best_model_path))
    print("ベストモデルの検証用データにおける損失：{}".format(checkpoint.best_model_score))

    test = trainer.test(test_dataloaders=dataloader_test)
    print("Test accuracy = {:.3f}".format(test[0]["test_acc"]))



if __name__ == "__main__":
    main()