import os

kftt_data_path = "./datasets/kftt-data-1.0/data/tok"

# data の中（2つともディレクトリ）
# orig, tok

# orig の中
# kyoto-dev.en  kyoto-dev.ja  kyoto-test.en  kyoto-test.ja  kyoto-train.en  kyoto-train.ja  kyoto-tune.en  kyoto-tune.ja

# tok の中
# train.cln は単語数 0 の文や非常に長い（40単語以上）の文を除いたデータ
# kyoto-dev.en  kyoto-dev.ja  kyoto-test.en  kyoto-test.ja  kyoto-train.cln.en  kyoto-train.cln.ja  kyoto-train.en  kyoto-train.ja  kyoto-tune.en  kyoto-tune.ja

# 今回はtok/kyoto-dev.en  kyoto-dev.ja  kyoto-test.en  kyoto-test.ja  kyoto-train.cln.en  kyoto-train.cln.ja を利用する。



def main():
    data_path = os.path.join(kftt_data_path, "./kyoto-train.cln.ja")
    print(data_path)
    with open(data_path, "r") as f:
        for line in f.readlines()[:3]:
            print(line)

    data_path = os.path.join(kftt_data_path, "./kyoto-train.cln.en")
    print(data_path)
    with open(data_path, "r") as f:
        for line in f.readlines()[:3]:
            print(line)

    for data in ["kyoto-train.cln.ja", "kyoto-train.cln.en", "kyoto-dev.ja", "kyoto-dev.en", "kyoto-test.ja", "kyoto-test.en"]:
        data_path = os.path.join(kftt_data_path, data)
        with open(data_path, 'r') as f:
            total_lines = sum(1 for _ in f)
        print("{} のデータサイズ：\t{}".format(data, total_lines))



if __name__ == "__main__":
    main()
