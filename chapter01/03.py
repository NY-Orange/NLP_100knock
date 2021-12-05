sentence = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
sentence = sentence.replace(",", "")
sentence = sentence.replace(".", "")
print(sentence)
word_list = sentence.split(" ")
for word in word_list:
    print(len(word))