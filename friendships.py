def import_friendships(file_name='friendships.txt'):
    friendships = {}

    current = None
    with open(file_name, 'r') as file:
        lines = file.readlines()

        for line in lines:
            if line.startswith('user'):
                current = line.split()[1].lower()
            elif line.startswith('friends'):
                if not current:
                    raise RuntimeError('Encountered friends before user.')

                # Like with the current user, all names are in lowercase
                friends = [friend.lower() for friend in line.split()[1:]]

                friendships[current] = friends

    return friendships
