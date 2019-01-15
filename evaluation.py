import pickle

from loguru import logger

from data_loader import import_data, import_results
from sentiment import predict, count_vectorizer, preprocessing, class_from_score


def calculate_would_buy():
    friendships, reviews = import_data()
    clusters, scores, purchased = import_results()

    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
        vocab = model['vocabulary']
        vocab_index = model['vocabulary_index']
        term_freq_matrix = model['term_frequency_per_class']
        num_terms_pr_class = model['terms_per_class']
        class_prob = model['class_probability']

    with open('communities.p', 'rb') as f:
        communities = pickle.load(f)

    logger.debug('Loaded data')

    # Replaces the review with a label.
    for user, review in reviews.items():
        data = preprocessing(review)
        vector = count_vectorizer(vocab, [data], vocab_index)[0]  # Count vectorizer expects and returns double array.
        reviews[user] = predict(vector, term_freq_matrix, num_terms_pr_class, class_prob)

    logger.debug('Predicted reviews')

    would_purchase = calculate_answer(communities, friendships, reviews)
    logger.debug('Calculated would purchase')

    _print_cluster_yes_percentage(communities, would_purchase)
    _print_cluster_acccuracy(communities, clusters)
    _print_review_accuracy(reviews, scores)
    _print_purchase_accuracy(would_purchase, purchased)

    with open('would_purchase.pkl', 'wb') as f:
        pickle.dump(would_purchase, f)


def calculate_answer(communities, friendships, reviews):
    would_purchase = {}

    # Calculate for each user if they would purchase a fine food
    for user, community in communities.items():
        count = 0
        score = 0

        # For each friend add the score and respective count
        for friend in friendships[user]:
            friend_community = communities[friend]

            # If friend has a review, get the label, otherwise continue.
            if friend in reviews:
                label = reviews[friend]
            else:
                continue

            # A friend outside the community counts for 10 times and kyle for 10 times
            if friend_community != community or friend == 'kyle':
                score += label * 10
                count += 10
            else:
                score += label
                count += 1

        # If there are no friends with review, the answer will be no. Otherwise it is calculated
        if count == 0:
            logger.debug(user)
            would_purchase[user] = 'no'
        else:
            would_purchase[user] = 'yes' if score / count >= 0.5 else 'no'

    return would_purchase


def _print_review_accuracy(our_guesses, dologs_guesses):
    count = 0
    skipped = 0

    # Calculates accuracy though we skip reviews with 3 as a score.
    for user, guess in our_guesses.items():
        _class = class_from_score(dologs_guesses[user])

        if _class:
            if guess == _class:
                count += 1
        else:
            skipped += 1

    print(f'Review accuracy: {count / (len(our_guesses) - skipped)}')


def _print_purchase_accuracy(our_guesses, dologs_guesses):
    count = 0
    for user, guess in our_guesses.items():
        if guess == dologs_guesses[user]:
            count += 1

    print(f'Purchase accuracy: {count / len(our_guesses)}')


def _print_cluster_acccuracy(our_guesses, dologs_guesses):
    cluster_conversion_index = {}

    count = 0
    for user, guess in our_guesses.items():
        if guess not in cluster_conversion_index:
            cluster_conversion_index[guess] = dologs_guesses[user]

        guess = cluster_conversion_index[guess]

        if guess == dologs_guesses[user]:
            count += 1

    print(f'Cluster accuracy: {count / len(our_guesses)}')


def _print_cluster_yes_percentage(communities, would_purchase):
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
