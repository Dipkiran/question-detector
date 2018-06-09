import nltk
from difflib import SequenceMatcher

text = open(file='question.txt', mode='r', encoding='utf8').read()
lowertext = text.lower()
sentence = lowertext.split("\n")
print(len(sentence))



sentences = []
for i in range(0,len(sentence)):
    new = sentence[i:]
    for sent in new:
        ratio = SequenceMatcher(None, sentence[i], sent).ratio()
        if (ratio > 0.5):
            newtext = sent.replace(sent, "")
            text = newtext
        else:
            sentences.append(sent)
            sentence = sentences
print(len(sentence))
    # print(i)
    # print(sentence[:i])
    # print(SequenceMatcher(None, sentence[i], sentence[:i]).ratio())