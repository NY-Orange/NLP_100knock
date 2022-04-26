import os
import sentencepiece as spm

output_data_path = "./outputs/ex95_outputs/"



def main():
    sp = spm.SentencePieceProcessor()

    spm_file = "ex95_output_spm.en"
    output_file = "ex95_output.en"

    sp.Load("./spm_models/kftt_spm_en_model.model")

    spm_file_path = os.path.join(output_data_path, spm_file)
    output_file_path = os.path.join(output_data_path, output_file)

    with open(spm_file_path, "r") as fr_spm, open(output_file_path, "w") as fw_output:
        for line in fr_spm:
            line = line.split(" ")
            line = "".join(sp.DecodePieces(line))
            fw_output.write(line)



if __name__ == "__main__":
    main()