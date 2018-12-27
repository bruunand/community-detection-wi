import numpy as np
from loguru import logger
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

import random


def _class_from_score(score):
    score_classes = {
        '1.0': 0,
        '2.0': 0,
        '3.0': None,
        '4.0': 1,
        '5.0': 1,
    }

    return score_classes[score]


def load_data(file_name):
    x, y = [], []

    current_class = None

    with open(file_name, 'r') as file:
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


if __name__ == '__main__':
    classes = {0, 1}
    train_x, train_y = load_data('SentimentTrainingData.txt')
    test_x, test_y = load_data('SentimentTestingData.txt')
    logger.debug('Loaded training and testing data')

    # Undersample
    train_x, train_y = _undersample(train_x, train_y)
    test_x, test_y = _undersample(test_x, test_y)

    # Some statistics on the data
    logger.debug(f'Training class distribution: {Counter(train_y)}')
    logger.debug(f'Testing class distribution: {Counter(test_y)}')

    # Fit transform count vectorizer on training data
    count_vectorizer = CountVectorizer()
    train_x = count_vectorizer.fit_transform(train_x)

    # Transform testing data with the fitted count vectorizer
    test_x = count_vectorizer.transform(test_x)

    # Train Naive Bayes classifier on data
    bayes = MultinomialNB()
    bayes.fit(train_x, train_y)
    logger.debug('Model fitted to data')

    # Evaluate
    predicted = bayes.predict(test_x)
    logger.info(f'Accuracy: {np.mean(predicted == test_y) * 100}%')
    logger.info(metrics.classification_report(test_y, predicted))
