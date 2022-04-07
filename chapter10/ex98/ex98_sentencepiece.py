import os
import sentencepiece as spm

JPara_data_path = "./datasets/en-ja"



def main():
    sp = spm.SentencePieceProcessor()
    spm.SentencePieceTrainer.Train("--input={} --model_prefix=./spm_models/JPara_spm_ja_model --vocab_size=32000".format(os.path.join(JPara_data_path, "./JPara-train-5000000.ja")))
    spm.SentencePieceTrainer.Train("--input={} --model_prefix=./spm_models/JPara_spm_en_model --vocab_size=32000".format(os.path.join(JPara_data_path, "./JPara-train-5000000.en")))

    for orig_file, spm_file in zip(
        ["JPara-train-5000000.ja", "JPara-train-5000000.en"],
        ["JPara-train-5000000-spm.ja", "JPara-train-5000000-spm.en"]
        ):

        sp.Load("./spm_models/JPara_spm_ja_model.model") if orig_file.endswith("ja") else sp.Load("./spm_models/JPara_spm_en_model.model")

        orig_file_path = os.path.join(JPara_data_path, orig_file)
        spm_file_path = os.path.join(JPara_data_path, "./spm_tok/"+spm_file)
        with open(orig_file_path, "r") as fr_orig, open(spm_file_path, "w") as fw_spm:
            for line in fr_orig:
                line = " ".join(sp.EncodeAsPieces(line))
                fw_spm.write(line + "\n")



if __name__ == "__main__":
    main()