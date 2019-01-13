import pickle

from loguru import logger

from data_loader import import_data
from sentiment import _predict, _count_vectorizer, preprocessing


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

    for user, review in reviews.items():
        data = preprocessing(review)
        vector = _count_vectorizer(vocab, [data], vocab_index)[0]
        reviews[user] = _predict(vector, term_freq_matrix, num_terms_pr_class, class_prob)

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
            if friend == 'kyle':
                if friend_community != community:
                    score += label * 100
                    count += 100
                else:
                    score += label * 10
                    count += 10
            elif friend_community != community:
                score += label * 10
                count += 10
            else:
                score += label
                count += 1

        # If there are no friends with review, the answer will be no. Otherwise it is calculated.
        if count == 0:
            would_purchase[user] = 'no'
        else:
            would_purchase[user] = 'yes' if score / count >= 0.5 else 'no'

    logger.debug('Calculated would purchase')

    with open('would_purchase.pkl', 'wb') as f:
        pickle.dump(would_purchase, f)


if __name__ == "__main__":
    calculate_would_buy()