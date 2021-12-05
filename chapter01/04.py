sentence = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
sentence = sentence.replace(".", "")
print(sentence)

word_list = sentence.split(" ")
print("", len(word_list), " words")

word_dic = dict()
atama = list()
key = ""
cnt = 1
for word in word_list:
    if cnt == 1 or cnt == 5 or cnt == 6 or cnt == 7 or cnt == 8 or cnt == 9 or cnt == 15 or cnt == 16 or cnt == 19:
        key = word[0]
    else:
        key = word[0] + word[1]
    word_dic[key] = cnt
    cnt += 1
print(word_dic)
print(cnt)