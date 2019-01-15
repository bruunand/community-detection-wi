import io


def import_data(file_name='data.txt'):
    friendships = {}
    reviews = {}

    current = None
    with io.open(file_name, mode='r', encoding='utf-8') as file:
        for line in file.readlines():
            if line.startswith('user'):
                current = line[6:].strip()
            else:
                if current:
                    if line.startswith('friends'):
                        # Strip any newline characters from the friend
                        friends = [friend.strip() for friend in line.split('\t')[1:]]

                        friendships[current] = friends
                    elif line.startswith('review'):
                        review = line[8:].strip()  # Skip review header and space

                        if review != '*':
                            reviews[current] = review

    return friendships, reviews


def import_results(file_name='friendships.reviews.results.txt'):
    clusters = {}
    scores = {}
    purchased = {}

    current = None
    with io.open(file_name, mode='r', encoding='utf-8') as file:
        for line in file.readlines():
            if line.startswith('user'):
                current = line[6:].strip()
            else:
                if current:
                    if line.startswith('cluster'):
                        # Strip any newline characters from the cluster
                        cluster = int(line[9:].strip())

                        clusters[current] = cluster
                    elif line.startswith('score'):
                        # Ensures string form of 'x.0', where x is a number between 1 and 5.
                        score = f'{line[7:].strip()}.0'

                        scores[current] = score

                    elif line.startswith('purchase'):
                        purchase = line[10:].strip()

                        purchased[current] = purchase

    return clusters, scores, purchased
