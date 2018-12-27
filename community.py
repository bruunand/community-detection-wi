import math
import pickle
import random

import numpy as np

from friendships import import_data
import networkx as nx

class Communities:
    def __init__(self):
        self.friendships = import_data()
        self.friendships_lst = list(self.friendships.items())
        self.edge_matrix = self.make_matrix()

    def spectral(self):
        print('Calculating laplacian')
        l = self.unnormalized_l()

        print('Calculating eigenvector')
        eigen = np.linalg.eig(l)[0].real.tolist()
        e_dict = {}

        # Create tuple of index and eigen value
        for index in range(len(eigen)):
            e_dict[index] = eigen[index]

        print('Starting making all communities')

        # Sorts eigenvector after eigen value
        eigen = sorted(e_dict.items(), key=lambda k: float(k[1]))

        communities = [eigen]
        community_store = [communities.copy()]

        # Split on highst difference until each vertices is in its own community or have made 15 splits.
        while len(communities) != len(self.friendships_lst) and len(community_store) < 15:
            communities = self.split_highest(communities)
            community_store.append(communities.copy())  # Save all splits.

        best = {'index': -1, 'modularity': -1}

        print('Calculating modularity for all communities')

        num_edges = self.calc_num_edges()

        # Calculates modularity for each community splitting in store,
        for i in range(len(community_store)):
            print(f'Modularity: {i / len(community_store) * 100}% processed')
            modularity = self.calc_modularity(community_store[i], num_edges)

            if modularity > best['modularity']:
                best['index'] = i
                best['modularity'] = modularity

        for community in community_store[best['index']]:
            print([self.friendships_lst[index][0] for index, _ in community])

    def calc_modularity(self, communities, num_edges):
        modularity = 0

        # Calculates the modularity of the communities.
        for community in communities:
            e = self.edges_contained_in_community(community) / num_edges  # probability and edge is in the community.
            a = self.edges_with_vertice_in_community(community) / num_edges  # probability a random edge would fall into the community.

            modularity += e - math.pow(a, 2)

        return modularity

    def calc_num_edges(self):
        sum = 0
        length = len(self.edge_matrix[0])
        for i in range(length):
            # Skip j < i as this is already seen as the matrix contains undirected edges.
            for j in range(length)[i:]:
                if self.edge_matrix[i][j] == 1:
                    sum += 1

        return sum

    def edges_contained_in_community(self, community):
        num = 0
        length = len(self.edge_matrix[0])

        # For constant lookup time
        in_com = {}
        for index, _ in community:
            in_com[index] = True

        # Count number of edges in community where both vertices are in the community.
        for index, _ in community:
            edges = self.edge_matrix[index]

            # Skips j < index to ensure an edge is only added once.
            for j in range(length)[index:]:
                if edges[j] == 1 and j in in_com:
                    num += 1

        return num

    def edges_with_vertice_in_community(self, community):
        num = 0
        length = len(self.edge_matrix[0])

        # Coun number of edges where one vertice is in the community.
        for index, _ in community:
            edges = self.edge_matrix[index]

            # Skips j < index to ensure an edge is only added once.
            for j in range(length)[index:]:
                if edges[j] == 1:
                    num += 1

        return num

    def unnormalized_l(self):
        d = self.make_degree(self.edge_matrix, len(self.edge_matrix[0]))

        return d - self.edge_matrix

    def split_highest(self, communities: list([])):
        split = {'diff': -1, 'position': None}

        # Goes through all communities and find the biggest differences
        for i in range(len(communities)):
            community = communities[i]
            for j in range(len(community))[1:]:
                diff = abs(community[j][1] - community[j-1][1])

                # Find biggest difference.
                if diff > split['diff']:
                    split['diff'] = diff
                    split['position'] = [(i, j)]
                # Add position if equal.
                elif diff == split['diff']:
                    split['position'].append((i, j))

        # randomly choose a position of the biggest differences to split on.
        position_i, position_j = random.choice(split['position'])

        # Generate new community splitting.
        new_c = []
        for i in range(len(communities)):
            if i == position_i:
                new_c.append(communities[i][:position_j])
                new_c.append(communities[i][position_j:])
            else:
                new_c.append(communities[i])

        return new_c

    @staticmethod
    def make_degree(matrix, length):
        # Calculate the out degree for each vertice.
        degree = np.zeros((length, length))
        for i in range(length):
            degree[i][i] = np.sum(matrix[i])

        return degree

    def make_matrix(self):
        length = len(self.friendships_lst)

        index_dict = self.create_index_dict(self.friendships_lst)

        matrix = np.zeros((length, length))

        # Make a matrix given the list, such that row 'i' in the matrix corresponds to the i'th element in the list.
        for i in range(length):
            friends = self.friendships_lst[i][1]
            length_friends = len(friends)
            for j in range(length_friends):
                if friends[j] not in index_dict:
                    continue

                index = index_dict[friends[j]]
                matrix[i][index] = 1
                matrix[index][i] = 1

        return matrix

    @staticmethod
    def create_index_dict(friendships: list):
        index_dict = {}

        for i in range(len(friendships)):
            index_dict[friendships[i][0]] = i

        return index_dict


c = Communities()
# c.make_graph(c.friendships)
c.spectral()
# print(c.calc_num_edges())


# print(c.split_highest(tmp))