import numpy as np
import networkx as nx
from networkx.algorithms import tree
from factor import Factor
from factor_utils import factor_product


""" ADD HELPER FUNCTIONS HERE (IF NEEDED) """


""" END ADD HELPER FUNCTIONS HERE """


def _get_clique_factors(jt_cliques, factors):
    """
    Assign node factors to cliques in the junction tree and derive the clique factors.

    Args:
        jt_cliques: list of junction tree maximal cliques e.g. [[x1, x2, x3], [x2, x3], ... ]
        factors: list of factors from the original graph

    Returns:
        list of clique factors where the factor(jt_cliques[i]) = clique_factors[i]
    """
    clique_factors = [Factor() for _ in jt_cliques]

    """ YOUR CODE HERE """
    del_list = []
    for i in range(len(jt_cliques)):
        joint = Factor()
        ic = jt_cliques[i]
        for j in range(len(factors)):
            if j not in del_list:
                if set(factors[j].var) <= set(ic):
                    joint = factor_product(joint, factors[j])
                    del_list.append(j)
        clique_factors[i] = joint
    
    pass




    """ END YOUR CODE HERE """

    assert len(clique_factors) == len(jt_cliques), 'there should be equal number of cliques and clique factors'
    return clique_factors


def _get_jt_clique_and_edges(nodes, edges):
    """
    Construct the structure of the junction tree and return the list of cliques (nodes) in the junction tree and
    the list of edges between cliques in the junction tree. [i, j] in jt_edges means that cliques[j] is a neighbor
    of cliques[i] and vice versa. [j, i] should also be included in the numpy array of edges if [i, j] is present.
    You can use nx.Graph() and nx.find_cliques().

    Args:
        nodes: numpy array of nodes [x1, ..., xN]
        edges: numpy array of edges e.g. [x1, x2] implies that x1 and x2 are neighbors.

    Returns:
        list of junction tree cliques. each clique should be a maximal clique. e.g. [[X1, X2], ...]
        numpy array of junction tree edges e.g. [[0,1], ...], [i,j] means that cliques[i]
            and cliques[j] are neighbors.
    """
    jt_cliques = []
    jt_edges = np.array(edges)  # dummy value

    """ YOUR CODE HERE """
    g1 = nx.Graph()
    g1.add_nodes_from(list(nodes))
    g1.add_edges_from(edges)
    # connected_subgraphs = list(nx.connected_components(g1))
    jt_cliques = list(nx.find_cliques(g1))
    # if len(connected_subgraphs) > 1:
    #     # Connect the unconnected components
    #     for i in range(1, len(connected_subgraphs)):
    #         u = min(connected_subgraphs[0])
    #         v = min(connected_subgraphs[i])
    #         g1.add_edge(u, v)


    g2 = nx.make_max_clique_graph(g1)
    jt_edges = np.array(g2.edges)
    g3 = nx.Graph()
    g3.add_nodes_from(g2.nodes)
    for i in range(len(jt_edges)):
        i1 = jt_edges[i][0]
        i2 = jt_edges[i][1]
        g3.add_edge(i1, i2, weight = len(set(jt_cliques[i1]).intersection(set(jt_cliques[i2]))))
    
    connected_subgraphs = list(nx.connected_components(g3))
    if len(connected_subgraphs) > 1:
        # Connect the unconnected components
        for i in range(1, len(connected_subgraphs)):
            u = min(connected_subgraphs[0])
            v = min(connected_subgraphs[i])
            g3.add_edge(u, v, weight = 0)    

    jt_edges = list(nx.maximum_spanning_tree(g3).edges)
    jt_m = [list(i) for i in jt_edges]
    jt_edges = jt_m[:]
    """ END YOUR CODE HERE """

    return jt_cliques, jt_edges


def construct_junction_tree(nodes, edges, factors):
    """
    Constructs the junction tree and returns its the cliques, edges and clique factors in the junction tree.
    DO NOT EDIT THIS FUNCTION.

    Args:
        nodes: numpy array of random variables e.g. [X1, X2, ..., Xv]
        edges: numpy array of edges e.g. [[X1,X2], [X2,X1], ...]
        factors: list of factors in the graph

    Returns:
        list of cliques e.g. [[X1, X2], ...]
        numpy array of edges e.g. [[0,1], ...], [i,j] means that cliques[i] and cliques[j] are neighbors.
        list of clique factors where jt_cliques[i] has factor jt_factors[i] where i is an index
    """
    jt_cliques, jt_edges = _get_jt_clique_and_edges(nodes=nodes, edges=edges)
    jt_factors = _get_clique_factors(jt_cliques=jt_cliques, factors=factors)
    return jt_cliques, jt_edges, jt_factors
