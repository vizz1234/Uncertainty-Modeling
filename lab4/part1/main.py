""" CS5340 Lab 4 Part 1: Importance Sampling
See accompanying PDF for instructions.

Name: Vishwanath Dattatreya Doddamani
Email: e1237250@u.nus.edu
Student ID: A0286188L
"""

import os
import json
import numpy as np
import networkx as nx
from factor_utils import factor_evidence, factor_marginalize, factor_product, assignment_to_index
from factor import Factor
from argparse import ArgumentParser
from tqdm import tqdm
import copy

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
INPUT_DIR = os.path.join(DATA_DIR, 'inputs')
PREDICTION_DIR = os.path.join(DATA_DIR, 'predictions')


""" ADD HELPER FUNCTIONS HERE """
def generate_graph(p_factors):
    """
    Creates Graph given the dictionary of proposal factors

    Args:
        p_factors: Dictionary of proposal factors
    
    Returns:
        Graph with nodes and edges corresponding to the proposal factors
    """

    G = nx.DiGraph()

    for key, value in p_factors.items():
        G.add_node(key)
        p_factors_var = value.var

        for rv in p_factors_var:
            if rv!=key:
                G.add_edge(rv, key)
    
    return G


""" END HELPER FUNCTIONS HERE """


def _sample_step(nodes, proposal_factors):
    """
    Performs one iteration of importance sampling where it should sample a sample for each node. The sampling should
    be done in topological order.

    Args:
        nodes: numpy array of nodes. nodes are sampled in the order specified in nodes
        proposal_factors: dictionary of proposal factors where proposal_factors[1] returns the
                sample distribution for node 1

    Returns:
        dictionary of node samples where samples[1] return the scalar sample for node 1.
    """
    samples = {}

    """ YOUR CODE HERE: Use np.random.choice """
    for node in nodes:
        factor = proposal_factors[node]
        factor = factor_evidence(factor, samples)
        factor_vars = factor.var
        ind = np.where(factor_vars == node)[0][0]
        card = factor.card[ind]
        prob = factor.val / np.sum(factor.val)
        sample = np.random.choice(card, size = card, p=prob)
        samples[node] = sample[0]

    """ END YOUR CODE HERE """

    assert len(samples.keys()) == len(nodes)
    return samples


def _get_conditional_probability(target_factors, proposal_factors, evidence, num_iterations):
    """
    Performs multiple iterations of importance sampling and returns the conditional distribution p(Xf | Xe) where
    Xe are the evidence nodes and Xf are the query nodes (unobserved).

    Args:
        target_factors: dictionary of node:Factor pair where Factor is the target distribution of the node.
                        Other nodes in the Factor are parent nodes of the node. The product of the target
                        distribution gives our joint target distribution.
        proposal_factors: dictionary of node:Factor pair where Factor is the proposal distribution to sample node
                        observations. Other nodes in the Factor are parent nodes of the node
        evidence: dictionary of node:val pair where node is an evidence node while val is the evidence for the node.
        num_iterations: number of importance sampling iterations

    Returns:
        Approximate conditional distribution of p(Xf | Xe) where Xf is the set of query nodes (not observed) and
        Xe is the set of evidence nodes. Return result as a Factor
    """
    out = Factor()

    """ YOUR CODE HERE """

    for key in proposal_factors.keys():
        proposal_factors[key] = factor_evidence(proposal_factors[key], evidence)
    
    for key in target_factors.keys():
        target_factors[key] = factor_evidence(target_factors[key], evidence)
    
    nodes_without_evd = np.setdiff1d(list(proposal_factors.keys()) , list(evidence.keys()))

    graph = generate_graph(proposal_factors)
    topological_nodes = list(nx.topological_sort(graph))
    topological_nodes = [n for n in topological_nodes if n in nodes_without_evd]
    topological_nodes = np.array(topological_nodes)

    nodes = np.sort(topological_nodes)

    out.var = nodes
    out.card = np.zeros(len(nodes))

    total_card = 1
    cum_card = []

    for i, key in enumerate(nodes):
        c1 = proposal_factors[key].card
        v1 = np.where(proposal_factors[key].var == key)[0][0]
        total_card = total_card * c1[v1]
        out.card[i] = c1[v1]
        cum_card.append(c1[v1])


    wl_dic = [[] for _ in range(total_card)]
    p_dic = [[] for _ in range(total_card)]
    out.val = np.zeros(total_card)

    for _ in tqdm(range(num_iterations)):
        

        sam_val = _sample_step(topological_nodes, proposal_factors)

        px_sam_val = {**sam_val, **evidence}

        pf_dup = Factor()
        tf_dup = Factor()

        p_xl = 1      
        
        for key in target_factors.keys():
            pf_dup = copy.deepcopy(target_factors[key])
            vars = pf_dup.var
            subset_dict = {k:v for k,v in px_sam_val.items() if k in vars}
            subset_keys = list(subset_dict.keys())
            subset_keys.sort()
            subset_values = [subset_dict[v] for v in subset_keys]
            idx = assignment_to_index(subset_values, pf_dup.card)
            p_xl = p_xl * pf_dup.val[idx]
        
        q_xl = 1

        for key in sam_val.keys():
            tf_dup = copy.deepcopy(proposal_factors[key])
            vars = tf_dup.var
            subset_dict = {k:v for k,v in px_sam_val.items() if k in vars}
            subset_keys = list(subset_dict.keys())
            subset_keys.sort()
            subset_values = [subset_dict[v] for v in subset_keys]
            idx = assignment_to_index(subset_values, tf_dup.card)
            q_xl = q_xl * tf_dup.val[idx]
            
        subset_keys = list(sam_val.keys())
        subset_keys.sort()
        subset_values = [sam_val[v] for v in subset_keys]
        idx = assignment_to_index(subset_values, cum_card)

        wl_dic[idx].append(p_xl / q_xl)
        p_dic[idx].append(p_xl * (p_xl / q_xl))
    
    for i in range(total_card):
        if len(wl_dic[i]) == 0:
            continue
        s_wl = np.sum(wl_dic[i])
        out.val[i] = np.sum(p_dic[i]) / s_wl
    
    out.val = out.val / np.sum(out.val)
    

    """ END YOUR CODE HERE """

    return out


def load_input_file(input_file: str) -> (Factor, dict, dict, int):
    """
    Returns the target factor, proposal factors for each node and evidence. DO NOT EDIT THIS FUNCTION

    Args:
        input_file: input file to open

    Returns:
        Factor of the target factor which is the target joint distribution of all nodes in the Bayesian network
        dictionary of node:Factor pair where Factor is the proposal distribution to sample node observations. Other
                    nodes in the Factor are parent nodes of the node
        dictionary of node:val pair where node is an evidence node while val is the evidence for the node.
    """
    with open(input_file, 'r') as f:
        input_config = json.load(f)
    target_factors_dict = input_config['target-factors']
    proposal_factors_dict = input_config['proposal-factors']
    assert isinstance(target_factors_dict, dict) and isinstance(proposal_factors_dict, dict)

    def parse_factor_dict(factor_dict):
        var = np.array(factor_dict['var'])
        card = np.array(factor_dict['card'])
        val = np.array(factor_dict['val'])
        return Factor(var=var, card=card, val=val)

    target_factors = {int(node): parse_factor_dict(factor_dict=target_factor) for
                      node, target_factor in target_factors_dict.items()}
    proposal_factors = {int(node): parse_factor_dict(factor_dict=proposal_factor_dict) for
                        node, proposal_factor_dict in proposal_factors_dict.items()}
    evidence = input_config['evidence']
    evidence = {int(node): ev for node, ev in evidence.items()}
    num_iterations = input_config['num-iterations']
    return target_factors, proposal_factors, evidence, num_iterations


def main():
    """
    Helper function to load the observations, call your parameter learning function and save your results.
    DO NOT EDIT THIS FUNCTION.
    """
    argparser = ArgumentParser()
    argparser.add_argument('--case', type=int, required=True,
                           help='case number to create observations e.g. 1 if 1.json')
    args = argparser.parse_args()
    # np.random.seed(0)

    case = args.case
    input_file = os.path.join(INPUT_DIR, '{}.json'.format(case))
    target_factors, proposal_factors, evidence, num_iterations = load_input_file(input_file=input_file)

    # solution part
    conditional_probability = _get_conditional_probability(target_factors=target_factors,
                                                           proposal_factors=proposal_factors,
                                                           evidence=evidence, num_iterations=num_iterations)
    print(conditional_probability)
    # end solution part

    # json only recognises floats, not np.float, so we need to cast the values into floats.
    save__dict = {
        'var': np.array(conditional_probability.var).astype(int).tolist(),
        'card': np.array(conditional_probability.card).astype(int).tolist(),
        'val': np.array(conditional_probability.val).astype(float).tolist()
    }

    if not os.path.exists(PREDICTION_DIR):
        os.makedirs(PREDICTION_DIR)
    prediction_file = os.path.join(PREDICTION_DIR, '{}.json'.format(case))

    with open(prediction_file, 'w') as f:
        json.dump(save__dict, f, indent=1)
    print('INFO: Results for test case {} are stored in {}'.format(case, prediction_file))


if __name__ == '__main__':
    main()
