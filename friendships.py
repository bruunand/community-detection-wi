def import_data(file_name='data.txt'):
    friendships = {}
    reviews = {}

    current = None
    with open(file_name, 'r') as file:
        for line in file.readlines():
            if line.startswith('user'):
                current = line.split()[1].lower()
            else:
                if current:
                    if line.startswith('friends'):
                        # Like with the current user, all names are in lowercase
                        friends = [friend.lower() for friend in line.split()[1:]]

                        friendships[current] = friends
                    elif line.startswith('review'):
                        review = line[8:].strip()  # Skip review header and space

                        if review != '*':
                            reviews[current] = review

    return friendships, reviews
