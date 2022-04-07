import re
import matplotlib.pyplot as plt



def main():
    beam_widths = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    bleu_scores = list()
    for beam_width in beam_widths:
        with open("./outputs/ex94_outputs/ex94_output_BLUE_SCORE_beam{}.txt".format(beam_width), "r") as f:
            pattern = re.compile(r'BLEU4 = (\d{2}\.\d{2})')
            bleu_score = pattern.search(f.read())
            print("beam_width: {}\tbleu_score: {}".format(beam_width, bleu_score.group(1)))
            bleu_scores.append(float(bleu_score.group(1)))

    fig = plt.figure()
    plt.plot(beam_widths, bleu_scores)
    plt.title("change in BLEU SCORE")
    plt.xlabel("beam width")
    plt.ylabel("BLEU SCORE")
    plt.xticks(beam_widths)
    plt.grid()
    fig.savefig("./outputs/ex94_outputs/ex94_output_BLEU_SCORES.png")



if __name__ == "__main__":
    main()