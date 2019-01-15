import pickle

from loguru import logger

from data_loader import import_data, import_results
from sentiment import predict, preprocessing, class_from_score


def calculate_would_buy():
    friendships, reviews = import_data()
    clusters, scores, purchased = import_results()

    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
        vocab_index = model['vocabulary_index']
        term_probability_matrix = model['term_probability_matrix']
        class_prob = model['class_probability']

    with open('community_detection/communities_test.p', 'rb') as f:
        communities = pickle.load(f)

    logger.debug('Loaded data')

    # Replaces the review with a label.
    for user, review in reviews.items():
        data = preprocessing(review)
        reviews[user] = predict(data, vocab_index, term_probability_matrix, class_prob)

    logger.debug('Predicted reviews')

    would_purchase = calculate_answer(communities, friendships, reviews)
    logger.debug('Calculated would purchase')

    print_cluster_yes_percentage(communities, would_purchase)
    print_cluster_accuracy(communities, clusters)
    print_review_accuracy(reviews, scores)
    print_review_precision(reviews, scores, 1)
    print_review_recall(reviews, scores, 1)
    print_review_precision(reviews, scores, 0)
    print_review_recall(reviews, scores, 0)
    print_purchase_accuracy(would_purchase, purchased)

    with open('would_purchase.pkl', 'wb') as f:
        pickle.dump(would_purchase, f)


def calculate_answer(communities, friendships, reviews):
    would_purchase = {}

    # Calculate for each user if they would purchase a fine food
    for user, community in communities.items():

        score = 0

        if user in reviews:
            score = reviews[user]

        else:
            # For each friend add the score and respective count
            for friend in friendships[user]:
                friend_community = communities[friend]

                # If friend has a review, get the label, otherwise continue.
                if friend in reviews:
                    label = reviews[friend]
                else:
                    continue

                # A friend outside the community counts for 10 times and kyle for 10 times
                weight = 1 if label else -1
                if friend_community != community or friend == 'kyle':
                    score += weight * 100
                else:
                    score += weight

        would_purchase[user] = 'yes' if score > 0 else 'no'

    return would_purchase


def print_review_accuracy(our_guesses, dologs_guesses):
    count = 0
    skipped = 0

    # Calculates accuracy though we skip reviews with 3 as a score.
    for user, guess in our_guesses.items():
        _class = class_from_score(dologs_guesses[user])

        if _class is not None:
            if guess == _class:
                count += 1
        else:
            skipped += 1

    print(f'Review accuracy: {count / (len(our_guesses) - skipped)}')


def print_review_precision(our_guesses, dologs_guesses, label):
    true_positive = 0
    positive = 0

    # Calculates accuracy though we skip reviews with 3 as a score.
    for user, guess in our_guesses.items():
        _class = class_from_score(dologs_guesses[user])

        if _class is not None:
            if guess == label:
                positive += 1
                if guess == _class:
                    true_positive += 1

    print(f'Review precision for {label}: {true_positive / positive}')


def print_review_recall(our_guesses, dologs_guesses, label):
    true_positive = 0
    all_positive = 0

    # Calculates accuracy though we skip reviews with 3 as a score.
    for user, guess in our_guesses.items():
        _class = class_from_score(dologs_guesses[user])

        if _class is not None:
            if _class == label:
                all_positive += 1
                if guess == _class:
                    true_positive += 1

    print(f'Review recall for {label}: {true_positive / all_positive}')


def print_purchase_accuracy(our_guesses, dologs_guesses):
    count = 0
    for user, guess in our_guesses.items():
        if guess == dologs_guesses[user]:
            count += 1

    print(f'Purchase accuracy: {count / len(our_guesses)}')


def print_cluster_accuracy(our_guesses, dologs_guesses):
    cluster_conversion_index = {}

    count = 0
    for user, guess in our_guesses.items():
        if guess not in cluster_conversion_index:
            cluster_conversion_index[guess] = dologs_guesses[user]

        guess = cluster_conversion_index[guess]

        if guess == dologs_guesses[user]:
            count += 1

    print(f'Cluster accuracy: {count / len(our_guesses)}')


def print_cluster_yes_percentage(communities, would_purchase):
    community_yes = {}

    # Calculate the percentage of yes's in each community
    for user, answer in would_purchase.items():
        community = communities[user]

        if community not in community_yes:
            community_yes[community] = {'yes_count': 0, 'size': 0}

        community_yes[community]['size'] += 1

        if answer == 'yes':
            community_yes[community]['yes_count'] += 1

    for community, answers in community_yes.items():
        yes_count = answers['yes_count']
        size = answers['size']
        print(f'Community {community}: {yes_count/size}')


if __name__ == "__main__":
    calculate_would_buy()
