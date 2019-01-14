import pickle

from loguru import logger

from data_loader import import_data
from sentiment import predict, count_vectorizer, preprocessing


def calculate_would_buy():
    friendships, reviews = import_data()

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

    # Estimate the review label.
    for user, review in reviews.items():
        data = preprocessing(review)
        vector = count_vectorizer(vocab, [data], vocab_index)[0]
        reviews[user] = predict(vector, term_freq_matrix, num_terms_pr_class, class_prob)

    logger.debug('Predicted reviews')

    would_purchase = {}

    # Calculate for each user if they would purchase a fine food.
    for user, community in communities.items():
        count = 0
        score = 0
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

        # If there are no friends with review, the answer will be no. Otherwise it is calculated.
        if count == 0:
            logger.debug(user)
            would_purchase[user] = 'no'
        else:
            would_purchase[user] = 'yes' if score / count >= 0.5 else 'no'

    logger.debug('Calculated would purchase')

    community_yes = {}
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
        print(f'{community}: {yes_count/size}')

    with open('would_purchase.pkl', 'wb') as f:
        pickle.dump(would_purchase, f)


if __name__ == "__main__":
    calculate_would_buy()
