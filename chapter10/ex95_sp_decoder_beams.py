import os
import sentencepiece as spm

output_data_path = "./outputs/ex95_outputs/"



def main():
    sp = spm.SentencePieceProcessor()

    sp.Load("./spm_models/kftt_spm_en_model.model")

    beam_widths = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90]

    for beam_width in beam_widths:
        spm_file = "ex95_output_beam{}_spm.en".format(beam_width)
        output_file = "ex95_output_beam{}.en".format(beam_width)

        spm_file_path = os.path.join(output_data_path, spm_file)
        output_file_path = os.path.join(output_data_path, output_file)

        with open(spm_file_path, "r") as fr_spm, open(output_file_path, "w") as fw_output:
            for line in fr_spm:
                line = line.split(" ")
                line = "".join(sp.DecodePieces(line))
                fw_output.write(line)



if __name__ == "__main__":
    main()