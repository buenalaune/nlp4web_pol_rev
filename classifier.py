import pandas as pd
from sklearn import metrics
import pickle



def load_data(test_file):
    test_data = pd.read_csv(test_file, sep=",", header=None)
    return test_data


def classify(test_data, model_name):
    vectorizer_name = './vectorizer.sav'
    vectorizer = pickle.load(open(vectorizer_name, 'rb'))
    classifier = pickle.load(open(model_name, 'rb'))
    test_tfidf_matrix = vectorizer.transform(test_data)
    prediction = classifier.predict(test_tfidf_matrix)
    return prediction


def calculate_domain(test_file):
    test_data = load_data(test_file)
    test_text, labels_test = test_data[0], test_data[1]
    model_name = './domain_classifier.sav'
    prediction = classify(test_text, model_name)
    print("Here is the accuracy for domain:", metrics.accuracy_score(prediction, labels_test) * 100, '%')


def calculate_polarity(test_file):
    test_data = load_data(test_file)
    test_text, labels_test = test_data[0], test_data[2]
    model_name = './polarity_classifier.sav'
    prediction = classify(test_text, model_name)
    print("Here is the accuracy for polarity:", metrics.accuracy_score(prediction, labels_test) * 100, '%')
    print('Correctly classified sentences:')

    pos_correct = 0
    neg_correct = 0
    for i in range(len(test_text)):
        if prediction[i] == labels_test[i]:
            print(test_text[i], "prediction", "positive" if prediction[i] == 3  else "negative")
            if labels_test[i] == 3:
                pos_correct += 1
            else:
                neg_correct += 1

    print('Number of correctly classified positive sentences:', pos_correct)
    print('Number of correctly classified negative sentences:', neg_correct)



if __name__ == '__main__':

    #test = "evaluation_examples_word_ex_simple.csv"
    test = "out.csv"
    calculate_domain(test)
    calculate_polarity(test)
