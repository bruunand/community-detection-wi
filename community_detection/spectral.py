import pickle

import scipy

import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from data_loader import import_data


def make_laplacian(friendships, friend_idx_dict):
    """ Construct unnormalized Laplacian matrix by L = D - A"""
    D = make_degree_matrix(friendships, friend_idx_dict)
    A = make_adjacency_matrix(friendships, friend_idx_dict)

    return D - A


def make_degree_matrix(friendships, friend_idx_dict):
    """ Returns the degree matrix given the friendships """
    degrees = []
    for i in range(len(friendships)):
        # The degree of a person is their number of friends
        degrees.append(len(friendships[friend_idx_dict[i]]))

    return np.diag(degrees)


def make_adjacency_matrix(friendships, idx_friend_dict):
    A = np.zeros((len(friendships), len(friendships)))

    # We need an inverse idx_friend dict for this one
    # In order to get the index given a friend
    friend_idx_dict = {v: k for k, v in idx_friend_dict.items()}

    for i in range(len(friendships)):
        friends = friendships[idx_friend_dict[i]]
        friends_idx = [friend_idx_dict[friend] for friend in friends]

        # For each friend, mark the corresponding cell with a 1
        for friend_idx in friends_idx:
            A[i][friend_idx] = 1

    return A


def get_friendships():
    friendships, _ = friendships, _ = import_data()
    return friendships


def get_idx_friend_dict(friendships):
    idx_to_friend = {}

    idx_counter = 0
    for friend in friendships:
        idx_to_friend[idx_counter] = friend
        idx_counter += 1

    return idx_to_friend


def run_spectral():
    friendships = get_friendships()
    idx_friend_dict = get_idx_friend_dict(friendships)
    logger.info(idx_friend_dict)

    # Make degree matrix
    L = make_laplacian(friendships, idx_friend_dict)

    # Find eigenvalues and eigenvectors
    # eigh returns eigenvalues and eigenvectors sorted by eigenvalue
    # This way we can get the second eigenvector, e.g. the one with the second smallest value
    eig_values, eig_vectors = scipy.linalg.eigh(L)

    # The second eigenvector is currently a column, we need to transpose it to use it in k-means
    second_vector = eig_vectors[:, 1]

    # Perform k-means on the eigenvector values
    kmeans = KMeans(n_clusters=4).fit(second_vector.reshape(-1, 1))
    person_cluster_dict = dict()
    for idx, label in enumerate(kmeans.labels_):
        logger.info(f'{idx_friend_dict[idx]} has label {label}')

        person_cluster_dict[idx_friend_dict[idx]] = label

    # Write communities to a file
    pickle.dump(person_cluster_dict, open('communities.p', 'wb'))


if __name__ == "__main__":
    run_spectral()

    # plt.plot(sorted(res))
    # plt.show()

