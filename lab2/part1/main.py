""" CS5340 Lab 2 Part 1: Junction Tree Algorithm
See accompanying PDF for instructions.

Name: Vishwanath Dattatreya Doddamani
Email: e1237250@u.nus.edu
Student ID: A0286188L
"""

import os
import numpy as np
import json
import networkx as nx
import copy
from argparse import ArgumentParser

from factor import Factor
from jt_construction import construct_junction_tree
from factor_utils import factor_product, factor_evidence, factor_marginalize

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
INPUT_DIR = os.path.join(DATA_DIR, 'inputs')  # we will store the input data files here!
PREDICTION_DIR = os.path.join(DATA_DIR, 'predictions')  # we will store the prediction files here!


""" ADD HELPER FUNCTIONS HERE """


""" END HELPER FUNCTIONS HERE """


def _update_mrf_w_evidence(all_nodes, evidence, edges, factors):
    """
    Update the MRF graph structure from observing the evidence

    Args:
        all_nodes: numpy array of nodes in the MRF
        evidence: dictionary of node:observation pairs where evidence[x1] returns the observed value of x1
        edges: numpy array of edges in the MRF
        factors: list of Factors in teh MRF

    Returns:
        numpy array of query nodes
        numpy array of updated edges (after observing evidence)
        list of Factors (after observing evidence; empty factors should be removed)
    """

    query_nodes = all_nodes
    updated_edges = edges
    updated_factors = []

    """ YOUR CODE HERE """
    for i in range(len(factors)):
        for key, value in evidence.items():
            if key in factors[i].var:
                if len(factors[i].var)>1:
                    factors[i] = factor_evidence(factors[i], {key:value})
                else:
                    factors[i] = Factor()
    for factor in factors:
        if factor.is_empty() == False:
            updated_factors.append(factor)

    ek = evidence.keys()  
    del_list = []  
    for i in range(len(edges)):
        for j in edges[i]:
            if j in ek:
                del_list.append(i)
                break
    updated_edges = np.delete(edges, del_list, axis = 0)
    query_nodes = query_nodes[~np.isin(query_nodes, list(evidence.keys()))]





    """ END YOUR CODE HERE """

    return query_nodes, updated_edges, updated_factors


def _get_clique_potentials(jt_cliques, jt_edges, jt_clique_factors):
    """
    Returns the list of clique potentials after performing the sum-product algorithm on the junction tree

    Args:
        jt_cliques: list of junction tree nodes e.g. [[x1, x2], ...]
        jt_edges: numpy array of junction tree edges e.g. [i,j] implies that jt_cliques[i] and jt_cliques[j] are
                neighbors
        jt_clique_factors: list of clique factors where jt_clique_factors[i] is the factor for cliques[i]

    Returns:
        list of clique potentials computed from the sum-product algorithm
    """
    clique_potentials = copy.deepcopy(jt_clique_factors)

    factors = copy.deepcopy(jt_clique_factors)

    jt_nodes = np.arange(0, len(jt_cliques), 1)

    if len(jt_nodes) == 1:
        return factors

    marginals = []

    graph = nx.Graph()

    graph.add_nodes_from(jt_nodes)

    graph.add_edges_from(jt_edges)

    # Setting up messages which will be passed
    # graph = generate_graph_from_factors(factors)

    # Uncomment the following line to visualize the graph. Note that we create
    # an undirected graph regardless of the input graph since 1) this
    # facilitates graph traversal, and 2) the algorithm for undirected and
    # directed graphs is essentially the same for tree-like graphs.
    # visualize_graph(graph)

    # You can use any node as the root since the graph is a tree. For simplicity
    # we always use node 0 for this assignment.

    #identifying root nodes
    
    # rn = []
    # for w in range(len(factors)):
    #     if len(factors[w].var) == 1:
    #         rn.append(factors[w].var[0])
    
    root = jt_nodes[0]


    # Create structure to hold messages
    num_nodes = len(jt_nodes)
    messages = [0 for _ in range(num_nodes)]
    M = np.array([[None]*num_nodes for _ in range(num_nodes)])
    Mf = [0 for _ in range(num_nodes)]
    m_fac = [1 for _ in range(num_nodes)]

    parent = {}
    child = {}
    child1 = {root:[]}
    k_list = list(jt_nodes)
    k_list.remove(root)
    child1.update({k:[] for k in k_list})
    # child = child1.copy()
    count = 0

    #identifying the children of nodes
    child[root] = list(graph.neighbors(root))
    parent[root] = []
    lp = 1
    for i in range(len(child[root])):
        parent[child[root][i]] = root
        lp = lp+1

    #identifying parent of nodes
    while len(parent) < num_nodes:
        for (key,value) in child1.copy().items():
            ck = [x for x in child.keys()]
            if key in child.keys():
                val = [v for v in child[key]]
                for c in val:
                    if c not in ck:
                        child[c] = list(graph.neighbors(c))
                        child[c].remove(parent[c])
                        if len(child[c]) == 0:
                            child[c] == []
                        for p in child[c]:
                            if p not in parent.keys():
                                parent[p] = c
                                lp = lp + 1

    if len(child) < len(child1):
        for ky, vl in child1.items():
            if ky not in child.keys():
                child[ky] = vl

    #identifying leaf nodes
    leaf_node = [x for (x,v) in child.items() if child[x]==[]]
    mc = []
    for leaf in leaf_node:
        key1 = leaf
        f = Factor()
        f = factors[leaf]  
        messages[key1] = Factor()
        p1 = set(factors[parent[leaf]].var)
        l1 = set(factors[leaf].var)
        fM = list(l1 - p1)
        messages[key1] = factor_marginalize(f, fM)
        M[key1, parent[key1]] = messages[key1]
        mc.append(key1)
        count=count+1
        f = None
    #Upward Pass
    while count < num_nodes:
        for (key,value) in child.items():
            co = 0
            if key not in mc:
                for v in value:
                    if v not in mc:
                        co=1
                        break

                if co==1:
                    continue
                ml = [k for k in value]
                joint = None

                joint = messages[ml[0]]

                for i in range(1,len(ml)):
                    joint = factor_product(joint,messages[ml[i]])
                
                if key!=root:
                    joint = factor_product(joint, factors[key])
                    p1 = set(factors[parent[key]].var)
                    l1 = set(factors[key].var)
                    fM = list(l1 - p1)
                    messages[key] = factor_marginalize(joint, fM)
                    # joint = factor_product(joint, graph.edges[parent[key],key]['factor'])
                    # joint = factor_marginalize(joint, [key])
                    M[key, parent[key]] = copy.deepcopy(messages[key])

                if key == root:
                    joint = factor_product(joint, copy.deepcopy(factors[key]))                
                    messages[key] = joint
                    Mf[key] = joint
                    # Mf[key].val = np.around(Mf[key].val/np.sum(Mf[key].val),8)
                    m_fac[key] = copy.deepcopy(factors[key])
                    clique_potentials[key] = messages[key]
                    for v1 in value:
                        joint1 = Factor()

                        indi1 = ml[:]
                        indi1.remove(v1)
                        joint1 = m_fac[key]
                        for j1 in indi1:
                            joint1 = factor_product(joint1, M[j1, key])
                        # joint1 = factor_product(joint1, graph.edges[v1, key]['factor'])

                        # joint1 = factor_marginalize(joint1, [key])
                        p1 = set(factors[v1].var)
                        l1 = set(joint1.var)
                        fM = list(l1 - p1)                        
                        joint1 = factor_marginalize(joint1, fM)                        
                        M[key,v1] = joint1
                        joint1 = Factor()

                joint = None
                mc.append(key)
                count = count+1
    
    #Downward Pass
    fV = []
    MD = [root]
    count = 1
    while count < num_nodes:
        for (key,value) in child.items():
            if key in MD:
                continue
            mld = child[key]
            for v2 in child[key]:
                # if key in rn:
                #     joint2 = copy.deepcopy(graph.nodes[key]['factor'])
                #     joint2 = factor_product(joint2, M[parent[key],key])
                # else:
                #     joint2 = copy.deepcopy(M[parent[key],key])
                joint2 = Factor()
                joint2 = M[parent[key],key]
                joint2 = factor_product(joint2, factors[key])
                indi1 = mld[:]
                indi1.remove(v2)
                for j1 in indi1:
                    joint2 = factor_product(joint2, M[j1, key])
                # joint2 = factor_product(joint2, graph.edges[v2, key]['factor'])
                p1 = set(joint2.var)
                l1 = set(factors[v2].var)
                fM = list(p1 - p1.intersection(l1))
                joint2 = factor_marginalize(joint2, fM)
                M[key,v2] = joint2
                joint2 = Factor()
            MD.append(key)
            count = count + 1
    
    #calculating marginals
    for i in range(num_nodes):
        if i!=root:
            mdp = M[:,i]
            joint3 = Factor()
            for j in mdp:
                if type(j) == Factor:
                    joint3 = factor_product(joint3, j)
            joint3 = factor_product(joint3, factors[i])
            # joint3.val = np.around(joint3.val/np.sum(joint3.val),8)
            Mf[i] = joint3
            clique_potentials[i] = joint3
            joint3 = Factor()
                    
    # fV = np.array(Mf)[V]
    # marginals = list(fV)

    # return marginals

    """ END YOUR CODE HERE """

    assert len(clique_potentials) == len(jt_cliques)
    return clique_potentials


def _get_node_marginal_probabilities(query_nodes, cliques, clique_potentials):
    """
    Returns the marginal probability for each query node from the clique potentials.

    Args:
        query_nodes: numpy array of query nodes e.g. [x1, x2, ..., xN]
        cliques: list of cliques e.g. [[x1, x2], ... [x2, x3, .., xN]]
        clique_potentials: list of clique potentials (Factor class)

    Returns:
        list of node marginal probabilities (Factor class)

    """
    query_marginal_probabilities = [Factor() for _ in range(len(query_nodes))]

    """ YOUR CODE HERE """
    for i in range(len(query_nodes)):
        for j in range(len(clique_potentials)):
            if query_nodes[i] in cliques[j]:
                marV = list(set(cliques[j]) - set([query_nodes[i]]))
                query_marginal_probabilities[i] = factor_marginalize(clique_potentials[j],marV)
                query_marginal_probabilities[i].val = query_marginal_probabilities[i].val / np.sum(query_marginal_probabilities[i].val)
                break


    """ END YOUR CODE HERE """
    # query_marginal_probabilities = clique_potentials

    return query_marginal_probabilities


def get_conditional_probabilities(all_nodes, evidence, edges, factors):
    """
    Returns query nodes and query Factors representing the conditional probability of each query node
    given the evidence e.g. p(xf|Xe) where xf is a single query node and Xe is the set of evidence nodes.

    Args:
        all_nodes: numpy array of all nodes (random variables) in the graph
        evidence: dictionary of node:evidence pairs e.g. evidence[x1] returns the observed value for x1
        edges: numpy array of all edges in the graph e.g. [[x1, x2],...] implies that x1 is a neighbor of x2
        factors: list of factors in the MRF.

    Returns:
        numpy array of query nodes
        list of Factor
    """
    query_nodes, updated_edges, updated_node_factors = _update_mrf_w_evidence(all_nodes=all_nodes, evidence=evidence,
                                                                              edges=edges, factors=factors)

    jt_cliques, jt_edges, jt_factors = construct_junction_tree(nodes=query_nodes, edges=updated_edges,
                                                               factors=updated_node_factors)

    clique_potentials = _get_clique_potentials(jt_cliques=jt_cliques, jt_edges=jt_edges, jt_clique_factors=jt_factors)

    query_node_marginals = _get_node_marginal_probabilities(query_nodes=query_nodes, cliques=jt_cliques,
                                                            clique_potentials=clique_potentials)

    return query_nodes, query_node_marginals


def parse_input_file(input_file: str):
    """ Reads the input file and parses it. DO NOT EDIT THIS FUNCTION. """
    with open(input_file, 'r') as f:
        input_config = json.load(f)

    nodes = np.array(input_config['nodes'])
    edges = np.array(input_config['edges'])

    # parse evidence
    raw_evidence = input_config['evidence']
    evidence = {}
    for k, v in raw_evidence.items():
        evidence[int(k)] = v

    # parse factors
    raw_factors = input_config['factors']
    factors = []
    for raw_factor in raw_factors:
        factor = Factor(var=np.array(raw_factor['var']), card=np.array(raw_factor['card']),
                        val=np.array(raw_factor['val']))
        factors.append(factor)
    return nodes, edges, evidence, factors


def main():
    """ Entry function to handle loading inputs and saving outputs. DO NOT EDIT THIS FUNCTION. """
    argparser = ArgumentParser()
    argparser.add_argument('--case', type=int, required=True,
                           help='case number to create observations e.g. 1 if 1.json')
    args = argparser.parse_args()

    case = args.case
    input_file = os.path.join(INPUT_DIR, '{}.json'.format(case))
    nodes, edges, evidence, factors = parse_input_file(input_file=input_file)

    # solution part:
    query_nodes, query_conditional_probabilities = get_conditional_probabilities(all_nodes=nodes, edges=edges,
                                                                                 factors=factors, evidence=evidence)

    predictions = {}
    for i, node in enumerate(query_nodes):
        probability = query_conditional_probabilities[i].val
        predictions[int(node)] = list(np.array(probability, dtype=float))

    if not os.path.exists(PREDICTION_DIR):
        os.makedirs(PREDICTION_DIR)
    prediction_file = os.path.join(PREDICTION_DIR, '{}.json'.format(case))
    with open(prediction_file, 'w') as f:
        json.dump(predictions, f, indent=1)
    print('INFO: Results for test case {} are stored in {}'.format(case, prediction_file))


if __name__ == '__main__':
    main()
