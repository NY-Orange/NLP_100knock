import os
import sentencepiece as spm

kftt_data_path = "./datasets/kftt-data-1.0/data/orig"



def main():
    sp = spm.SentencePieceProcessor()
    spm.SentencePieceTrainer.Train("--input={} --model_prefix=./spm_models/kftt_spm_ja_model --vocab_size=24000".format(os.path.join(kftt_data_path, "./kyoto-train.ja")))
    spm.SentencePieceTrainer.Train("--input={} --model_prefix=./spm_models/kftt_spm_en_model --vocab_size=24000".format(os.path.join(kftt_data_path, "./kyoto-train.en")))

    for orig_file, spm_file in zip(
        ["kyoto-train.ja", "kyoto-train.en", "kyoto-dev.ja", "kyoto-dev.en", "kyoto-test.ja", "kyoto-test.en"],
        ["kyoto-train-spm.ja", "kyoto-train-spm.en", "kyoto-dev-spm.ja", "kyoto-dev-spm.en", "kyoto-test-spm.ja", "kyoto-test-spm.en"]
        ):

        sp.Load("./spm_models/kftt_spm_ja_model.model") if orig_file.endswith("ja") else sp.Load("./spm_models/kftt_spm_en_model.model")

        orig_file_path = os.path.join(kftt_data_path, orig_file)
        spm_file_path = os.path.join(kftt_data_path, "./../spm_tok/"+spm_file)
        with open(orig_file_path, "r") as fr_orig, open(spm_file_path, "w") as fw_spm:
            for line in fr_orig:
                line = " ".join(sp.EncodeAsPieces(line))
                fw_spm.write(line + "\n")



if __name__ == "__main__":
    main()