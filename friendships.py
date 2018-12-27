def import_data(file_name='data.txt'):
    friendships = {}
    reviews = {}

    current = None
    with open(file_name, 'r') as file:
        for line in file.readlines():
            if line.startswith('user'):
                current = line.split()[1].lower()
            elif line.startswith('friends'):
                if not current:
                    raise RuntimeError('Encountered friends before user.')

                # Like with the current user, all names are in lowercase
                friends = [friend.lower() for friend in line.split()[1:]]

                friendships[current] = friends

    return friendships
