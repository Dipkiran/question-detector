import random
from sklearn.svm import SVC
from nltk.classify.scikitlearn import SklearnClassifier
import nltk
import pickle
import os
import time
import numpy as np
def qdataset(partition=0.8):
    text1 = open(file='question.txt', mode='r', encoding='utf8').read()
    text2 = open(file='answer.txt', mode='r', encoding='utf8').read()
    text = text1+text2
    sentence1 = text1.split("\n")
    sentence2 = text2.split("\n")
    sentences = sentence1 + sentence2
    se
    text = text.split()
    vocab = list(set(text))
    vocab = sorted(vocab)
    delimiter = ["how"]
    char_to_id = {ch: id for id, ch in enumerate(vocab)}
    id_to_char = {id: ch for id, ch in enumerate(vocab)}

    input = [[char_to_id[word]] for sent in sentences for word in sent.split()]

    # labels = [1 if item in delimiter else 0 for item in text]
    labels = [1 if item in word for sent in sentence1 for word in sent.split() else 0 for for sent in sentences for word in sent.split()  ]
    # label2 = [0 for sent in sentence2 for word in sent.split()]
    # labels = label1 + label2
    print(labels)


    partition = int(len(input) * partition)
    # print(len(input))
    train_input, train_labels, test_input, test_labels = input[:partition], labels[:partition], \
                                                         input[partition:], labels[partition:]

    return(train_input, train_labels, test_input, test_labels, char_to_id, id_to_char)
#train_input, train_labels, test_input, test_labels, char_to_id, id_to_char
def classifier(REPORT_ACCURACY=True, classifier_file="question.tct"):
    train_input, train_labels, test_input, test_labels, char_to_id, id_to_char = qdataset()

    print(len(train_input))
    print(len(train_labels))
    print(len(test_input))
    print(len(test_labels))
    classifier = SVC()
    classifier.fit(X=train_input, y=train_labels)
    if REPORT_ACCURACY:
        score = classifier.score(X=test_input, y=test_labels)
        print('Accuracy : ', "%.2f" % score)


qdataset()
classifier()
