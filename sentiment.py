from loguru import logger
from collections import Counter


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
    x, y = ([] for _ in range(2))

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


if __name__ == '__main__':
    train_x, train_y = load_data('SentimentTrainingData.txt')
    test_x, test_y = load_data('SentimentTestingData.txt')
    logger.debug('Loaded training and testing data')

    # Some statistics on the data
    logger.debug(f'Training class distribution: {Counter(train_y).values()}')
    logger.debug(f'Testing class distribution: {Counter(test_y).values()}')
