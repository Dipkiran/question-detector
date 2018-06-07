import random
from sklearn.svm import SVC
from nltk.classify.scikitlearn import SklearnClassifier
import nltk
import pickle
import os
import time
import numpy as np
def qdataset(partition=0.8):

    #taking input from two files: question and answer
    text1 = open(file='question.txt', mode='r', encoding='utf8').read()
    text2 = open(file='answer.txt', mode='r', encoding='utf8').read()

    #combining two list
    text = text1+text2

    #sorting sentences from text and combining them
    sentence1 = text1.split()
    sentence2 = text2.split("\n")
    sentences = sentence1 + sentence2

    #finding number of words in a sentence
    text = text.split()
    vocab = list(set(text))
    vocab = sorted(vocab)

    #change char into some id and viceversa
    char_to_id = {ch: id for id, ch in enumerate(vocab)}
    id_to_char = {id: ch for id, ch in enumerate(vocab)}

    #replace word in sentence by some id
    input = [[char_to_id[word]] for sent in sentences for word in sent.split()]

    #assign 0 for answer and 1 for question
    label1 = [1 for sent in sentence1 for word in sent.split()]
    label2 = [0 for sent in sentence2 for word in sent.split()]
    labels = label1 + label2

    #shuffling the data randomly
    new = list(zip(input, labels))
    random.shuffle(new)
    input, labels = zip(*new)

    #diving data into test and train data
    partition = int(len(input) * partition)
    train_input, train_labels, test_input, test_labels = input[:partition], labels[:partition], \
                                                         input[partition:], labels[partition:]

    return(train_input, train_labels, test_input, test_labels, char_to_id, id_to_char)

def classifier(REPORT_ACCURACY=True):
    train_input, train_labels, test_input, test_labels, char_to_id, id_to_char = qdataset()


    classifier = SVC()
    classifier.fit(X=train_input, y=train_labels)
    if REPORT_ACCURACY:
        score = classifier.score(X=test_input, y=test_labels)
        print('Accuracy : ', "%.2f" % score)


classifier()
