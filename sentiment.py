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

stop_words = set(stopwords.words('english'))


def class_from_score(score):
    score_classes = {
        '0.0': 0,
        '1.0': 0,
        '2.0': 0,
        '3.0': 1,
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
                current_class = class_from_score(split[1])
            elif split[0] == 'review/text' and current_class is not None:
                x.append(split[1])
                y.append(current_class)

                current_class = None

    return x, y


def shuffle_lists(*ls):
    zipped = list(zip(*ls))
    random.shuffle(zipped)
    return zip(*zipped)


def _undersample(x, y, random_state=None):
    if random_state:
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


def naive_bayes():
    classes = {0, 1}
    train_x, train_y = load_sentiment_data('SentimentTrainingData.txt')
    test_x, test_y = load_sentiment_data('SentimentTestingData.txt')
    logger.debug('Loaded training and testing data')

    # Undersample
    train_x, train_y = _undersample(train_x, train_y)
    test_x, test_y = _undersample(test_x, test_y)

    # Preprocess both train and test data
    train_x = [preprocessing(text) for text in train_x]
    test_x = [preprocessing(text) for text in test_x]
    logger.debug('Preprocessed training and test data')

    # Create vocabulary
    vocabulary = create_vocabulary(train_x)

    # Generate vocabulary index
    term_to_index = {}
    for i in range(len(vocabulary)):
        term_to_index[vocabulary[i]] = i

    logger.info('Created vocabulary')

    class_prob = {}
    num_data = len(train_y)

    # Calculate the probability of a class
    for cls in classes:
        count = train_y.count(cls)
        # Log space probability used to prevent underflow
        # Laplace smoothing
        class_prob[cls] = math.log((count + 1) / (num_data + len(classes)))
    logger.debug('Calculated class probabilities')

    # Count the number of occurrences for each term over all reviews
    # We will need to compute the probability of a word given a class
    term_freq_matrix = count_term_occurrence(train_x, train_y, classes, term_to_index)
    logger.debug('Calculated term count for each class')

    # Get the number of terms in each class
    terms_per_class = np.sum(term_freq_matrix, axis=0)

    # Calculate the probability of a term occurring given a class.
    term_probability_matrix = calculate_term_probabilities(term_freq_matrix, terms_per_class, len(vocabulary))
    logger.debug('Calculated term probability matrix')

    # Create term count vectors
    matrix = count_vectorizer(vocabulary, test_x, term_to_index)

    # Predict the class of test labels
    predictions = []
    for text in test_x:
        predictions.append(predict(text, term_to_index, term_probability_matrix, class_prob))

    logger.debug('Predicted classes on test set')

    # Compute various metrics
    acc, precision_pos, precision_neg, recall_pos, recall_neg = get_measures(predictions, test_y)

    # Save model parts
    with open('model.pkl', 'wb') as f:
        pickle.dump(
            {'vocabulary': vocabulary, 'vocabulary_index': term_to_index,
             'term_probability_matrix': term_probability_matrix, 'class_probability': class_prob}, f)

    # Save measures
    with open("results_no_under.pkl", 'wb') as f:
        pickle.dump({'accuracy': acc, 'precision_pos': precision_pos, 'recall_pos': recall_pos,
                     'precision_neg': precision_neg, 'recall_neg': recall_neg}, f)


def calculate_term_probabilities(term_freq_matrix, terms_per_class, vector_length):
    term_probability_matrix = np.zeros((len(term_freq_matrix), len(terms_per_class)))

    # Calculate the term probability for each term for a class.
    for term_index in range(len(term_freq_matrix)):
        for class_index in range(len(terms_per_class)):
            # Uses Laplace smoothing to ensure no log of 0
            term_probability_matrix[term_index][class_index] = math.log((term_freq_matrix[term_index][class_index] + 1)
                                                                        / (terms_per_class[class_index]
                                                                           + vector_length))

    return term_probability_matrix


def get_measures(predictions, labels):
    # Out of all of the model's prediction, how many did it get right?
    acc = len([1 for pred, label in zip(predictions, labels) if pred == label]) / len(labels)

    # Get the different measures.
    precision_pos = get_precision(predictions, labels, 1)
    precision_neg = get_precision(predictions, labels, 0)
    recall_pos = get_recall(predictions, labels, 1)
    recall_neg = get_recall(predictions, labels, 0)

    return acc, precision_pos, precision_neg, recall_pos, recall_neg


def get_precision(predictions, labels, _class):
    retrieved = 0
    true_guess = 0

    # Count all documents matching the class and count where these were correct
    for pred, label in zip(predictions, labels):
        if pred == _class:
            retrieved += 1

            if pred == label:
                true_guess += 1

    return true_guess / retrieved


def get_recall(predictions, labels, _class):
    relevant = 0
    true_guess = 0

    # Count all real matches with the class and count how many were correctly guessed
    for pred, label in zip(predictions, labels):
        if label == _class:
            relevant += 1

            if pred == label:
                true_guess += 1

    return true_guess / relevant


def create_vocabulary(reviews):
    vocab = set()

    # Create vocabulary of terms
    for review in reviews:
        vocab.update([term for term in review])

    return sorted(list(vocab))


def count_vectorizer(vocab, data, term_to_index):
    length = len(vocab)

    matrix = []

    # For each review count the term occurrence frequency
    for review in data:
        count_map = np.zeros((length,))
        for term in review:
            if term in term_to_index:
                count_map[term_to_index[term]] += 1

        matrix.append(count_map)

    return matrix


def count_term_occurrence(data, labels, classes, vocab_index: dict):
    vocab_length = len(vocab_index.items())
    matrix = np.zeros((vocab_length, len(classes)))

    # For each text/review count the number of occurrences for each class
    for text, label in zip(data, labels):
        for term in text:
            # Skips if not in dictionary
            if term not in vocab_index:
                continue

            index = vocab_index[term]

            matrix[index][label] += 1

    return matrix


def predict(text, term_to_index, term_probability_matrix, class_prob):
    class_scores = np.zeros((len(class_prob)))
    class_instances = len(class_prob)
    # For each term calculate the probability using log. Log is used to insure no underflow as
    # the standard formula Pi(p(x_i | c)) would be a small number
    for term in set(text):
        if term in term_to_index:
            term_index = term_to_index[term]
        else:
            continue

        for class_index in range(class_instances):
            # Skip term if no occurrences in vector
            class_scores[class_index] += term_probability_matrix[term_index][class_index]

    # Add the class probability
    for class_index in range(len(class_prob)):
        class_scores[class_index] += class_prob[class_index]

    class_scores = list(class_scores)

    # Return the label of the vector
    return class_scores.index(max(class_scores))


def preprocessing(text):
    """ Tokenizes a string, and NOP the tokens

    Arguments:
        string {str} -- A string of words.

    Returns:
        list -- A list containing stemmed tokens.
    """
    # Convert document to lowercase and replace apostrophes
    # Apostrophes are removed because Treebank style tokenization splits them from their word
    text = text.lower().replace('\'', '')

    # Remove HTML
    soup = bs4.BeautifulSoup(text, 'html.parser')
    text = soup.text

    return [token for token in word_tokenize(text) if re.match(r'\w+', token) and token not in stop_words]


if __name__ == "__main__":
    naive_bayes()
