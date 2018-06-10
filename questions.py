import random
from sklearn.svm import SVC
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
import numpy as np

from nltk.tokenize import wordpunct_tokenize
MAX_LEN = 15
def qdataset(partition=0.8):
    # taking input from two files: question.txt and answer.txt
    text1 = open(file='question.txt', mode='r', encoding='utf8').read()
    text2 = open(file='answer.txt', mode='r', encoding='utf8').read()

    # combining two list
    text = text1 + text2

    # sorting sentences from text and combining them
    sentence1 = text1.split("\n")
    sentence2 = text2.split("\n")
    sentences = sentence1 + sentence2
    # finding number of words in a sentence
    text = text.split()
    vocab = list(set(text))
    vocab = sorted(vocab)

    # change char into some id and viceversa
    char_to_id = {ch: id for id, ch in enumerate(vocab)}
    id_to_char = {id: ch for id, ch in enumerate(vocab)}

    input = []
    # replace word in sentence by some id
    for sent in sentences:
        vector = [char_to_id[word] for word in wordpunct_tokenize(sent)]
        padding = [-1]*MAX_LEN
        vector +=padding[:MAX_LEN-len(vector)]
        input.append(vector)




    # assign 0 for answer and 1 for question
    label1 = [1 for sent in sentence1]
    label2 = [0 for sent in sentence2]
    labels = label1 + label2


    # shuffling the data randomly
    new = list(zip(input, labels))
    random.shuffle(new)
    input, labels = zip(*new)

    # dividing data into test and train data
    partition = int(len(input) * partition)
    train_input, train_labels, test_input, test_labels = input[:partition], labels[:partition], \
                                                         input[partition:], labels[partition:]
    return train_input, train_labels, test_input, test_labels, char_to_id, id_to_char


def quesclassifier(REPORT_ACCURACY=True):

        # print("1. Prepare dataset")
        train_input, train_labels, test_input, test_labels, char_to_id, id_to_char = qdataset()
        # print('No of train samples {}, test samples {}'.format(len(train_labels), len(test_input)))

        # print("2. Initialize classifier and train")
        classifier = RandomForestClassifier()
        classifier.fit(X=train_input, y=train_labels)
        #
        # file = open(classifier_file, mode='wb')
        # pickle.dump(classifier, file, protocol=pickle.HIGHEST_PROTOCOL)
        # pickle.dump(char_to_id, file, protocol=pickle.HIGHEST_PROTOCOL)
        # pickle.dump(id_to_char, file, protocol=pickle.HIGHEST_PROTOCOL)

        if REPORT_ACCURACY:
            score = classifier.score(X=test_input, y=test_labels)
            print('Accuracy : ', "%.2f" % score)

    # else:
    #     print('Reusing pretrained classifier')
    #     file = open(classifier_file, mode='rb')
    #     classifier = pickle.load(file)
    #     char_to_id = pickle.load(file)
    #     id_to_char = pickle.load(file)

        return classifier, char_to_id, id_to_char


class QuestionDetector:
    """ check whether the given statement is question or not"""

    def __init__(self):
        self.classifier, self.char_to_id, self.id_to_char = quesclassifier()

    def question(self, text):
        """
                Algorithm:
                    Generates sentences from a text.
                    Classifier predicts if a sentence is a question or not.
                    print the output
        """
        # Check if user input is empty
        if text == '' or text is None:
            return text

        # split user text into sentences and compute sentid
        sentences = text.split()
        print(sentences)
        sentid = [self.char_to_id.get(word, -1) for word in sentences]
        padding = [-1] * MAX_LEN
        sentid += padding[:MAX_LEN - len(sentid)]
        # print(sentid)
        send = np.int32(sentid).reshape((1, -1))
        # print(send)

        isques = self.classifier.predict(send)
        print(isques)# predicts if a sentence is a question or not.

        if isques:
            print(sentences)  # if the text is the question
        # if text is not the question


if __name__ == '__main__':
    text = ['how are you','i am fine', 'what do you do', 'where are you from','my name is dip']
    # tokenizer = QuestionDetector()
    # tokenizer.question(text)

    for check in text:
        tokenizer = QuestionDetector()
        tokenizer.question(check)
