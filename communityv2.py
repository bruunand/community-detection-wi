import numpy as np

from friendships import import_data
import networkx as nx

class Communities:
    def __init__(self):
        self.friendships = import_data()

    def edge_to_remove(self, graph: nx.Graph):
        betweeness = nx.edge_betweenness_centrality(graph)
        worst = sorted(betweeness.items(), key=lambda k: k[1], reverse=True)[0][0]
        return worst

    def girvan(self):
        graph = self.make_graph(self.friendships)
        num_nodes = graph.number_of_nodes()
        c = nx.connected_component_subgraphs(graph)
        l = len(list(c))
        prev_l = l

        graph_store = [graph.copy()]

        while l != num_nodes:
            graph.remove_edge(*self.edge_to_remove(graph))
            c = nx.connected_component_subgraphs(graph)
            l = len(list(c))

            if l > prev_l:
                graph_store.append(graph.copy())
                prev_l = l

        modularities = []
        for index in range(len(graph_store)):
            modularity = self.calculate_modularity(graph_store[index])
            modularities.append((modularity, index))

        modularities.sort(key=lambda k: k[0], reverse=True)

        return graph_store[modularities[0][1]]



    def calculate_modularity(self, graph):
        sum = 0
        num_edges = graph.number_of_edges()
        for module in nx.connected_component_subgraphs(graph):
            e = self.edges_in_module(module) / num_edges
            a = self.edges_with_end_in_module(module) / num_edges
            sum += e - pow(a, 2)

        return sum

    def edges_in_module(self, graph: nx.Graph):
        nodes = dict(graph.nodes)
        counter = 0
        for v1, v2 in graph.edges:
            if v1 in nodes and v2 in nodes:
                counter += 1

        return counter

    def edges_with_end_in_module(self, graph: nx.Graph):
        nodes = dict(graph.nodes)
        counter = 0
        for v1, v2 in graph.edges:
            if v1 in nodes or v2 in nodes:
                counter += 1

        return counter




    @staticmethod
    def make_graph(friendships: dict):
        graph = nx.Graph()

        for user in list(friendships.keys())[:100]:
            graph.add_node(user)

        for user, friends in list(friendships.items())[:100]:
            [graph.add_edge(user, friend) for friend in friends[:10]]

        return graph

c = Communities()
c.girvan()