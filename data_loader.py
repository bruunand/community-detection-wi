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
