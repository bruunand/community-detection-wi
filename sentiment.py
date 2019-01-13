import io
import math
import pickle
import random
import re
from collections import Counter

import bs4
import numpy as np
from loguru import logger
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

stop_words = set(stopwords.words('english'))


def _class_from_score(score):
    score_classes = {
        '1.0': 0,
        '2.0': 0,
        '3.0': None,
        '4.0': 1,
        '5.0': 1,
    }

    return score_classes[score]


def load_sentiment_data(file_name):
    x, y = [], []

    current_class = None

    with io.open(file_name, mode='r', encoding='utf-8') as file:
        for line in file.readlines():
            split = [x.strip() for x in line.split(':')]

            if split[0] == 'review/score':
                current_class = _class_from_score(split[1])
            elif split[0] == 'review/text' and current_class is not None:
                x.append(split[1])
                y.append(current_class)

                current_class = None

    return x, y


def shuffle_lists(*ls):
    zipped = list(zip(*ls))
    random.shuffle(zipped)
    return zip(*zipped)


def _undersample(x, y, random_state=0):
    random.seed = random_state

    x, y = shuffle_lists(x, y)

    # Get min class count
    counter = Counter(y)
    min_count = min(counter.values())

    # Undersample by taking min_count items from every class
    ret_x, ret_y = [], []
    for cls in counter.keys():
        cls_idx = set([idx for idx, val in enumerate(y) if val == cls][:min_count])

        ret_x.extend([val for idx, val in enumerate(x) if idx in cls_idx])
        ret_y.extend([cls] * min_count)

    return shuffle_lists(ret_x, ret_y)


def _preprocess(corpus):
    """ Remove HTML and perform negation. """
    punctuation = re.compile('[.:;!?]')
    negatives = {'don\'t', 'never', 'nothing', 'nowhere', 'noone', 'none', 'not', 'no', 'hasn\'t', 'hadn\'t', 'can\'t',
                 'couldn\'t', 'shouldn\'t', 'won\'t', 'wouldn\'t', 'don\'t', 'doesn\'t', 'didn\'t', 'isn\'t', 'aren\'t',
                 'ain\'t'}

    ret_corpus = []

    for text in corpus:
        text = text.lower().split()  # Remove HTML and split newlines
        new_text = []

        negate = False
        for word in text:
            new_text.append(word if not negate else f'neg_{word}')

            if word in negatives:
                negate = True
            elif punctuation.findall(word):
                negate = False

        ret_corpus.append(' '.join(new_text))

    return ret_corpus


def _train_model():
    classes = {0, 1}
    train_x, train_y = load_sentiment_data('SentimentTrainingData.txt')
    test_x, test_y = load_sentiment_data('SentimentTestingData.txt')
    logger.debug('Loaded training and testing data')

    # Undersample
    #train_x, train_y = _undersample(train_x, train_y)
    #test_x, test_y = _undersample(test_x, test_y)

    # Perform negation on the input sets
    train_x = _preprocess(train_x)
    test_x = _preprocess(test_x)

    # Some statistics on the data
    logger.debug(f'Training class distribution: {Counter(train_y)}')
    logger.debug(f'Testing class distribution: {Counter(test_y)}')

    # Construct classifier as a pipeline
    sent_classifier = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('term-frequencyf', TfidfTransformer(use_idf=False)),
        ('classifier', MultinomialNB())
    ])
    sent_classifier.fit(train_x, train_y)
    logger.debug('Model fitted to data')

    # Evaluate
    predicted = sent_classifier.predict(test_x)
    logger.info(f'Accuracy: {np.mean(predicted == test_y) * 100}%')
    logger.info(metrics.classification_report(test_y, predicted))

    return sent_classifier


def _naive_bayes():
    classes = {0, 1}
    train_x, train_y = load_sentiment_data('SentimentTrainingData.txt')
    test_x, test_y = load_sentiment_data('SentimentTestingData.txt')
    logger.debug('Loaded training and testing data')

    # Undersample
    train_x, train_y = _undersample(train_x, train_y)
    test_x, test_y = _undersample(test_x, test_y)

    train_x = [preprocessing(text) for text in train_x]
    test_x = [preprocessing(text) for text in test_x]
    logger.debug('Preprocessed training and test data')

    # Create vocabulary
    vocab = _create_vocabulary(train_x)

    # Generate vocabulary index.
    vocab_index = {}
    for i in range(len(vocab)):
        vocab_index[vocab[i]] = i

    logger.info('Created vocabulary')

    # Create term count vectors.
    matrix = _count_vectorizer(vocab, test_x, vocab_index)

    class_prob = {}
    num_data = len(train_y)

    # Calculate the probability of a class.
    for _class in classes:
        count = train_y.count(_class)
        class_prob[_class] = math.log(count / num_data)  # Use log such that no underflow occur.
    logger.debug('Calculated class probability')

    # Count the number of occurrences for each term over all reviews.
    term_freq_matrix = _count_term_occurrence(train_x, train_y, classes, vocab_index)
    logger.debug('Calculated term count for each class')

    # Get the number of terms in each class.
    num_terms_pr_class = np.sum(term_freq_matrix, axis=0)

    # Predict the class of test labels
    predictions = []
    for i in range(len(matrix)):
        predictions.append(_predict(matrix[i], term_freq_matrix, num_terms_pr_class, class_prob))

    logger.debug('Predicted classes on test set')

    acc, precision_pos, precision_neg, recall_pos, recall_neg = _get_measures(predictions, test_y)

    with open("results_no_under.pkl", 'wb') as f:
        pickle.dump({'accuracy': acc, 'precision_pos': precision_pos, 'recall_pos': recall_pos,
                     'precision_neg': precision_neg, 'recall_neg': recall_neg}, f)

def _get_measures(predictions, labels):
    acc = len([1 for pred, label in zip(predictions, labels) if pred == label]) / len(labels)

    # Get the different measures.
    precision_pos = _get_precision(predictions, labels, 1)
    precision_neg = _get_precision(predictions, labels, 0)
    recall_pos = _get_recall(predictions, labels, 1)
    recall_neg = _get_recall(predictions, labels, 0)

    return acc, precision_pos, precision_neg, recall_pos, recall_neg


def _get_precision(predictions, labels, _class):
    retrieved = 0
    true_guess = 0

    # Count all documents matching the class and count where these were correct.
    for pred, label in zip(predictions, labels):
        if pred == _class:
            retrieved += 1

            if pred == label:
                true_guess += 1

    return  true_guess / retrieved


def _get_recall(predictions, labels, _class):
    relevant = 0
    true_guess = 0

    # Count all real matches with the class and count how many were correctly guessed.
    for pred, label in zip(predictions, labels):
        if label == _class:
            relevant += 1

            if pred == label:
                true_guess += 1

    return true_guess / relevant


def _create_vocabulary(reviews):
    vocab = set()

    # Create vocabulary of terms.
    for review in reviews:
        for term in review:
            vocab.add(term)

    return sorted(list(vocab))


def _count_vectorizer(vocab, data, vocab_index):
    length = len(vocab)

    matrix = []

    # For each review count the term occurrence frequency.
    for review in data:
        countmap = np.zeros((length,))
        for term in review:
            if term in vocab_index:
                countmap[vocab_index[term]] += 1

        matrix.append(countmap)

    return matrix


def _count_term_occurrence(data, labels, classes, vocab_index: dict):
    vocab_length = len(vocab_index.items())
    matrix = np.zeros((vocab_length, len(classes)))

    # for each text/review count the number of occurrences for each class.
    for text, label in zip(data, labels):
        for term in text:
            # Skips if not in dictionary.
            if term not in vocab_index:
                continue

            index = vocab_index[term]

            matrix[index][label] += 1

    return matrix


def _predict(vector, term_freq_matrix, num_terms_pr_class, class_prob):
    score = np.zeros((len(num_terms_pr_class)))
    vector_length = len(vector)  # The size of our vocabulary
    class_lenght = len(num_terms_pr_class)

    # For each term calculate the probability using log. Log is used to insure no underflow as
    # the standard formula Pi(p(x_i | c)) would be a small number. We therefore get the log probability,
    # we can compare on.
    for term_index in range(vector_length):
        for class_index in range(class_lenght):
            if vector[term_index] != 0:
                # Uses laplace to ensure no log of 0.
                score[class_index] += math.log((term_freq_matrix[term_index][class_index] + 1) /
                                               (num_terms_pr_class[class_index] + vector_length))

    # Add the class probability.
    for class_index in range(len(num_terms_pr_class)):
        score[class_index] += class_prob[class_index]

    score = list(score)

    return score.index(max(score))  # returns the label of the vector.


def preprocessing(text):
    """ Tokenizes a string, and NOP the tokens

    Arguments:
        string {str} -- A string of words.

    Returns:
        list -- A list containing stemmed tokens.
    """

    # Remove HTML
    soup = bs4.BeautifulSoup(text, 'html.parser')
    text = soup.text

    # Get all tokens that is not a stop word and only contains alphanumeric letters
    words = [token for token in word_tokenize(text) if token not in stop_words and re.match(r'\w+', token)]

    return words


def get_sentiments(reviews):
    sentiments = {}

    model = _train_model()
    for friend, review in reviews.items():
        sentiments[friend] = model.predict([review])[0]

    return sentiments


if __name__ == "__main__":
    _naive_bayes()
