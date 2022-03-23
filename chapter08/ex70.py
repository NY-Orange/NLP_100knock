import string
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors

# 問題50と同じ
def preprocess_dataset():
    df = pd.read_csv("./datasets/NewsAggregatorDataset/newsCorpora.csv", sep="\t", header=None, names=["ID", "TITLE", "URL", "PUBLISHER", "CATEGORY", "STORY", "HOSTNAME", "TIMESTAMP"])
    df = df.loc[df["PUBLISHER"].isin(["Reuters", "Huffington Post", "Businessweek", "Contactmusic.com", "Daily Mail"]), :]

    train, valid_test = train_test_split(df.loc[:,["TITLE", "CATEGORY"]], test_size=0.2, shuffle=True, stratify=df["CATEGORY"])
    valid, test = train_test_split(valid_test, test_size=0.5, shuffle=True, stratify=valid_test["CATEGORY"])

    print('【学習データ】')
    print(train['CATEGORY'].value_counts())

    return train, valid, test

def instance_to_vector(text, model):
    table = str.maketrans(string.punctuation, " "*len(string.punctuation))
    words = text.translate(table).split()
    vectors = [model[word] for word in words if word in model]

    return torch.tensor(sum(vectors) / len(vectors))



def main():
    train, valid, test = preprocess_dataset()

    # 問題60と同じ
    model = KeyedVectors.load_word2vec_format("./datasets/GoogleNews-vectors-negative300.bin.gz", binary=True)

    print("単語ベクトルの次元数：", len(model["United_States"]))
    print(train.iloc[:5])
    
    X_train = torch.stack([instance_to_vector(text, model) for text in train["TITLE"]])
    X_valid = torch.stack([instance_to_vector(text, model) for text in valid["TITLE"]])
    X_test = torch.stack([instance_to_vector(text, model) for text in test["TITLE"]])

    print("X_train_size : ", X_train.size())
    print("X_train :\n", X_train[:5])

    category_dict = {"b":0, "e":1, "t":2, "m":3}
    y_train = torch.tensor(train["CATEGORY"].map(lambda x: category_dict[x]).values)
    y_valid = torch.tensor(valid["CATEGORY"].map(lambda x: category_dict[x]).values)
    y_test = torch.tensor(test["CATEGORY"].map(lambda x: category_dict[x]).values)

    print("y_train_size : ", y_train.size())
    print("y_train :\n", y_train)

    # https://pytorch.org/docs/stable/generated/torch.save.html
    # 一般的なPyTorchの規約では、.pt という拡張子を使ってテンソルを保存する。
    torch.save(X_train, "./data/X_train.pt")
    torch.save(X_valid, "./data/X_valid.pt")
    torch.save(X_test, "./data/X_test.pt")
    torch.save(y_train, "./data/y_train.pt")
    torch.save(y_valid, "./data/y_valid.pt")
    torch.save(y_test, "./data/y_test.pt")



if __name__ == "__main__":
    main()