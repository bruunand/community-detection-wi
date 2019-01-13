import pickle

import networkx as nx
import matplotlib.pyplot as plt
from loguru import logger

from data_loader import import_data


class Communities:
    def __init__(self):
        self.friendships, _ = import_data()

    def edge_to_remove(self, graph: nx.Graph):
        # Calculate betweenness of edges
        betweenness = nx.edge_betweenness_centrality(graph)

        # Sort by highest betweenness and get the highest element edge
        # Intuitively, this edge has the highest amount of shortest paths from every point to every other point
        highest = sorted(betweenness.items(), key=lambda k: k[1], reverse=True)

        return highest[0][0]

    def run_girvan(self):
        # Get the graph
        graph = self.make_graph(self.friendships)

        # Get status of starting graph
        num_nodes = graph.number_of_nodes()
        components = nx.connected_component_subgraphs(graph)
        n_components = len(list(components))
        prev_l = n_components

        graph_store = [graph.copy()]

        # Continue removing edges until all edges are removed or there are 15 communities.
        # We stop at 5 as we expect between 2 and 10 communities.
        while n_components != num_nodes and n_components < 7:
            graph.remove_edge(*self.edge_to_remove(graph))
            logger.info('Removing edge')
            components = nx.connected_component_subgraphs(graph)
            n_components = len(list(components))

            # If edge removal resulted in more communities, add to store and update prev_l.
            if n_components > prev_l:
                graph_store.append(graph.copy())
                prev_l = n_components

                logger.info(f'{n_components} components')

        # Calculate modularity for all stored splits
        modularities = []
        for index in range(len(graph_store)):
            modularity = self.calculate_modularity(graph_store[index], graph_store[0])
            # Store modularity and index
            modularities.append((modularity, index))

        # Sort by highest modularity
        modularities.sort(key=lambda k: k[0], reverse=True)

        # Returns the graph with the best modularity
        return graph_store[modularities[0][1]]

    def calculate_modularity(self, graph, original_graph):
        e_sum = 0
        num_edges = graph.number_of_edges()
        # If there are no edges, the modularity is zero
        # May possibly be wrong
        if not num_edges:
            return 0

        # Calculate modularity by summing percent of edges in a module and percent of edges with one vertex in a module
        for module in nx.connected_component_subgraphs(graph):
            # There is an error when we are at the lowest level
            # There are no edges, so we get a division by zero error
            edges_in_module = self.edges_in_module(module, original_graph.edges) / num_edges
            end_in_module = self.edges_with_end_in_module(module, original_graph.edges) / num_edges
            e_sum += edges_in_module - pow(end_in_module, 2)

        return e_sum

    @staticmethod
    def edges_in_module(graph: nx.Graph, edges):
        nodes = dict(graph.nodes)
        counter = 0
        for v1, v2 in edges:
            if v1 in nodes and v2 in nodes:
                counter += 1

        return counter

    @staticmethod
    def edges_with_end_in_module(graph: nx.Graph, edges):
        nodes = dict(graph.nodes)
        counter = 0
        for v1, v2 in edges:
            if v1 in nodes or v2 in nodes:
                counter += 1

        return counter

    @staticmethod
    def make_graph(friendships: dict):
        graph = nx.Graph()

        for user in list(friendships.keys()):
            graph.add_node(user)

        for user, friends in list(friendships.items()):
            [graph.add_edge(user, friend) for friend in friends]

        return graph


if __name__ == "__main__":
    c = Communities()
    result = c.run_girvan()

    logger.info(f'{len(list(nx.connected_component_subgraphs(result)))} communities')
    pickle.dump(result, open('girvan.p', 'wb'))
    logger.info(result)
