import string
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split

# 問題50と同じ
def preprocess_dataset():
    df = pd.read_csv("./datasets/NewsAggregatorDataset/newsCorpora_re.csv", sep="\t", header=None, names=["ID", "TITLE", "URL", "PUBLISHER", "CATEGORY", "STORY", "HOSTNAME", "TIMESTAMP"])
    df = df.loc[df["PUBLISHER"].isin(["Reuters", "Huffington Post", "Businessweek", "Contactmusic.com", "Daily Mail"]), :]

    train, valid_test = train_test_split(df.loc[:,["TITLE", "CATEGORY"]], test_size=0.2, shuffle=True, stratify=df["CATEGORY"])
    valid, test = train_test_split(valid_test, test_size=0.5, shuffle=True, stratify=valid_test["CATEGORY"])

    return train, valid, test

def make_word2id_dict(newsdata):
    word_freq_dict = defaultdict(int)
    table = str.maketrans(string.punctuation, " "*len(string.punctuation))
    for text in newsdata["TITLE"]:
        for word in text.translate(table).split():
            word_freq_dict[word] += 1
    word_freq_dict = sorted(word_freq_dict.items(), key=lambda x:x[1], reverse=True)

    word2id_dict = {word: i+1 for i, (word, freq) in enumerate(word_freq_dict) if freq >= 2}

    return word2id_dict

def text2id(text, word2id_dict, unk=0):
    table = str.maketrans(string.punctuation, " "*len(string.punctuation))
    # 辞書の get メソッドの第2引数では key が存在しなかったときに返す値を指定
    return [word2id_dict.get(word, unk) for word in text.translate(table).split()]



def main():
    train, _, _ = preprocess_dataset()

    word2id_dict = make_word2id_dict(train)

    text = train.iloc[1, train.columns.get_loc("TITLE")]
    print("テキスト：{}".format(text))
    print("　ID列　：{}".format(text2id(text, word2id_dict)))



if __name__ == "__main__":
    main()