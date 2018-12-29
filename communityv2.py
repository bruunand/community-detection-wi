import numpy as np

from friendships import import_data
import networkx as nx

class Communities:
    def __init__(self):
        self.friendships, _ = import_data()

    def edge_to_remove(self, graph: nx.Graph):
        # Calculate betweenness of edges.
        betweenness = nx.edge_betweenness_centrality(graph)

        # Sort by highest betweenness and get the highest elements edge.
        highest = sorted(betweenness.items(), key=lambda k: k[1], reverse=True)[0][0]

        return highest

    def girvan(self):
        # Get the graph
        graph = self.make_graph(self.friendships)

        # Get status of starting graph
        num_nodes = graph.number_of_nodes()
        c = nx.connected_component_subgraphs(graph)
        l = len(list(c))
        prev_l = l

        graph_store = [graph.copy()]

        # Continue removing edges until all edges are removed or there are 15 communities.
        # We stop at 15 as we expect between 2 and 10 communities.
        while l != num_nodes and l < 15:
            graph.remove_edge(*self.edge_to_remove(graph))
            c = nx.connected_component_subgraphs(graph)
            l = len(list(c))

            # If edge removal resulted in more communities, add to store and update prev_l.
            if l > prev_l:
                graph_store.append(graph.copy())
                prev_l = l

        modularities = []

        # Calculate modularity for all stored splits.
        for index in range(len(graph_store)):
            modularity = self.calculate_modularity(graph_store[index], graph_store[0])
            modularities.append((modularity, index))        # Store modularity and index.

        # Sort by highest modularity.
        modularities.sort(key=lambda k: k[0], reverse=True)

        # Returns the graph with the best modularity.
        return graph_store[modularities[0][1]]

    def calculate_modularity(self, graph, original_graf):
        _sum = 0
        num_edges = graph.number_of_edges()

        # Calculate modularity by summing percent of edges in a module and percent of edges with one vertice in a module.
        for module in nx.connected_component_subgraphs(graph):
            e = self.edges_in_module(module, original_graf.edges) / num_edges
            a = self.edges_with_end_in_module(module, original_graf.edges) / num_edges
            _sum += e - pow(a, 2)

        return _sum

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
