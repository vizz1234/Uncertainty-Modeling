""" CS5340 Lab 1: Belief Propagation and Maximal Probability
See accompanying PDF for instructions.

Name: Vishwanath Dattatreya Doddamani
Email: e1237250@u.nus.edu
Student ID: A0286188L
"""

import copy
from typing import List

import numpy as np

from factor import Factor, index_to_assignment, assignment_to_index, generate_graph_from_factors, \
    visualize_graph


"""For sum product message passing"""
def factor_product(A, B):
    """Compute product of two factors.

    Suppose A = phi(X_1, X_2), B = phi(X_2, X_3), the function should return
    phi(X_1, X_2, X_3)
    """
    if A.is_empty():
        return B
    if B.is_empty():
        return A

    # Create output factor. Variables should be the union between of the
    # variables contained in the two input factors
    out = Factor()
    out.var = np.union1d(A.var, B.var)

    # Compute mapping between the variable ordering between the two factors
    # and the output to set the cardinality
    out.card = np.zeros(len(out.var), np.int64)
    mapA = np.argmax(out.var[None, :] == A.var[:, None], axis=-1)
    mapB = np.argmax(out.var[None, :] == B.var[:, None], axis=-1)
    out.card[mapA] = A.card
    out.card[mapB] = B.card

    # For each assignment in the output, compute which row of the input factors
    # it comes from
    out.val = np.zeros(np.prod(out.card))
    assignments = out.get_all_assignments()
    idxA = assignment_to_index(assignments[:, mapA], A.card)
    idxB = assignment_to_index(assignments[:, mapB], B.card)


    """ YOUR CODE HERE
    You should populate the .val field with the factor product
    Hint: The code for this function should be very short (~1 line). Try to
      understand what the above lines are doing, in order to implement
      subsequent parts.
    """
    out.val = np.multiply(A.val[idxA],B.val[idxB])


    ''' '''

    return out


def factor_marginalize(factor_1, var):
    """Sums over a list of variables.

    Args:
        factor_1 (Factor): Input factor
        var (List): Variables to marginalize out

    Returns:
        out: Factor with variables in 'var' marginalized out.
    """
    factor = copy.deepcopy(factor_1)

    out = Factor()

    """ YOUR CODE HERE
    Marginalize out the variables given in var
    """
    out.var = np.array(list(set(factor.var) - set(var)))

    ind = []


    for i in range(len(out.var)):
        i1 = np.where(factor.var == out.var[i])[0]
        out.card = np.append(out.card,factor.card[i1])

    var_mar_card = []
    for i in range(len(var)):
        i1 = np.where(factor.var == var[i])[0]
        var_mar_card.append(factor.card[i1])
    
  

    ass = factor.get_all_assignments()  
    
    for i in range(len(var)):
        i1 = np.where(factor.var == var[i])[0]
        i2 = var_mar_card[i]
        i3 = np.arange(0,i2[0],1)
    

        i4 = ass[:,i1]
        ind = []

        for j in i3:
            ind.append(np.where(i4.flatten() == j))
        
        true_ind = np.stack([l1 for l1 in ind], axis=-1)

        if len(true_ind) == 0:
            pass

        sum1 = 0
        
        del_list = []

        for l in range(len(true_ind[0])):
            sum1 = np.sum(factor.val[true_ind[0,l]])
            factor.val[true_ind[0,l][:]] = sum1
            del_list.extend(true_ind[0,l][1:])
        factor.val = np.delete(factor.val,del_list)
        ass = np.delete(ass,del_list,axis=0)

    out.val = factor.val

    return out


def observe_evidence(factors, evidence=None):
    """Modify a set of factors given some evidence

    Args:
        factors (List[Factor]): List of input factors
        evidence (Dict): Dictionary, where the keys are the observed variables
          and the values are the observed values.

    Returns:
        List of factors after observing evidence
    """
    if evidence is None:
        return factors
    out = copy.deepcopy(factors)

    """ YOUR CODE HERE
    Set the probabilities of assignments which are inconsistent with the 
    evidence to zero.
    """
    for (key,value) in evidence.items():
        var1 = key
        val1 = value
        for i in range(len(factors)):
            fac_var_i = factors[i].var
            if var1 in fac_var_i:
                ass_fac_var_i = factors[i].get_all_assignments()
                i1 = np.where(fac_var_i == var1)
                i2 = np.where(ass_fac_var_i[:,i1[0][0]] == val1)
                for j in range(len(factors[i].val)):
                    if j not in i2[0]:
                        out[i].val[j] = 0






    return out


"""For max sum meessage passing (for MAP)"""
def factor_sum(A, B):
    """Same as factor_product, but sums instead of multiplies
    """
    if A.is_empty():
        return B
    if B.is_empty():
        return A

    # Create output factor. Variables should be the union between of the
    # variables contained in the two input factors
    out = Factor()
    out.var = np.union1d(A.var, B.var)

    # Compute mapping between the variable ordering between the two factors
    # and the output to set the cardinality
    out.card = np.zeros(len(out.var), np.int64)
    mapA = np.argmax(out.var[None, :] == A.var[:, None], axis=-1)
    mapB = np.argmax(out.var[None, :] == B.var[:, None], axis=-1)
    out.card[mapA] = A.card
    out.card[mapB] = B.card

    # For each assignment in the output, compute which row of the input factors
    # it comes from
    out.val = np.zeros(np.prod(out.card))
    assignments = out.get_all_assignments()
    idxA = assignment_to_index(assignments[:, mapA], A.card)
    idxB = assignment_to_index(assignments[:, mapB], B.card)

    """ YOUR CODE HERE
    You should populate the .val field with the factor sum. The code for this
    should be very similar to the factor_product().
    """
    out.val = A.val[idxA] + B.val[idxB]


    return out


def factor_max_marginalize(factor_1, var):
    """Marginalize over a list of variables by taking the max.

    Args:
        factor_1 (Factor): Input factor
        var (List): Variable to marginalize out.

    Returns:
        out: Factor with variables in 'var' marginalized out. The factor's
          .val_argmax field should be a list of dictionary that keep track
          of the maximizing values of the marginalized variables.
          e.g. when out.val_argmax[i][j] = k, this means that
            when assignments of out is index_to_assignment[i],
            variable j has a maximizing value of k.
          See test_lab1.py::test_factor_max_marginalize() for an example.
    """
    out = Factor()

    """ YOUR CODE HERE
    Marginalize out the variables given in var. 
    You should make use of val_argmax to keep track of the location with the
    maximum probability.
    """
    factor = copy.deepcopy(factor_1)
    out.var = np.array(list(set(factor.var) - set(var)))
    ind = []
    for i in range(len(out.var)):
        i1 = np.where(factor.var == out.var[i])[0]
        out.card = np.append(out.card,factor.card[i1])

    var_mar_card = []
    for i in range(len(var)):
        i1 = np.where(factor.var == var[i])[0]
        var_mar_card.append(factor.card[i1])
  
    ass = factor.get_all_assignments()  
    
    for i in range(len(var)):
        i1 = np.where(factor.var == var[i])[0]
        i2 = var_mar_card[i]
        i3 = np.arange(0,i2[0],1)
        i4 = ass[:,i1]
        ind = []

        for j in i3:
            ind.append(np.where(i4.flatten() == j))
        
        true_ind = np.stack([l1 for l1 in ind], axis=-1)
        del_list = []

        for l in range(len(true_ind[0])):
            max_i = np.argmax(factor.val[true_ind[0,l]])
            dl = true_ind[0,l][:]
            dl = np.delete(dl,max_i)
            del_list.extend(dl)
        factor.val = np.delete(factor.val,del_list)
        ass = np.delete(ass,del_list,axis=0)
    
    
    out.val = factor.val

    indices = np.where(np.isin(factor.var, var))[0]

    out.val_argmax = [{} for _ in range(len(ass))]

    for r in range(len(ass)):
        vc = ass[r]
        for value in range(len(vc)):
            if value in indices:
                out.val_argmax[r][factor.var[value]] = vc[value] 
    
    #Reordering out.val according to factor assignment
    dce = []
    cova = [{} for _ in range(len(ass))] 
    for i in range(len(var)):
        dc = np.where(var[i] == factor_1.var)[0]
        dce.append(dc[0])
    ds = list(set(np.arange(0,len(factor_1.var),1)) - set(dce))
    oa = out.get_all_assignments()
    cov = [0.0] * len(out.val)
    for i in range(len(oa)):
        i1 = np.where(ass[:,ds][i] == oa)[0][0]
        cov[i1] = out.val[i]
        cova[i1] = out.val_argmax[i] 
    
    cov = np.array(cov)
    out.val = cov[:]
    out.val_argmax = cova[:]


    return out


def compute_joint_distribution(factors):
    """Computes the joint distribution defined by a list of given factors

    Args:
        factors (List[Factor]): List of factors

    Returns:
        Factor containing the joint distribution of the input factor list
    """
    joint = Factor()

    """ YOUR CODE HERE
    Compute the joint distribution from the list of factors. You may assume
    that the input factors are valid so no input checking is required.
    """
    joint = factors[0]

    for i in range(1,len(factors)):
        joint = factor_product(joint,factors[i])
        

    return joint


def compute_marginals_naive(V, factors, evidence):
    """Computes the marginal over a set of given variables

    Args:
        V (int): Single Variable to perform inference on
        factors (List[Factor]): List of factors representing the graphical model
        evidence (Dict): Observed evidence. evidence[k] = v indicates that
          variable k has the value v.

    Returns:
        Factor representing the marginals
    """

    output = Factor()

    """ YOUR CODE HERE
    Compute the marginal. Output should be a factor.
    Remember to normalize the probabilities!
    """
    output = compute_joint_distribution(factors)
    output = observe_evidence([output], evidence)
    mar_var = list(set(output[0].var) - set([V]))
    output = factor_marginalize(output[0], mar_var)
    output.val = output.val / sum(output.val)

    return output


def compute_marginals_bp(V, factors, evidence):
    """Compute single node marginals for multiple variables
    using sum-product belief propagation algorithm

    Args:
        V (List): Variables to infer single node marginals for
        factors (List[Factor]): List of factors representing the grpahical model
        evidence (Dict): Observed evidence. evidence[k]=v denotes that the
          variable k is assigned to value v.

    Returns:
        marginals: List of factors. The ordering of the factors should follow
          that of V, i.e. marginals[i] should be the factor for variable V[i].
    """
    # Dummy outputs, you should overwrite this with the correct factors
    marginals = []

    # Setting up messages which will be passed
    factors = observe_evidence(factors, evidence)
    graph = generate_graph_from_factors(factors)

    # Uncomment the following line to visualize the graph. Note that we create
    # an undirected graph regardless of the input graph since 1) this
    # facilitates graph traversal, and 2) the algorithm for undirected and
    # directed graphs is essentially the same for tree-like graphs.
    # visualize_graph(graph)

    # You can use any node as the root since the graph is a tree. For simplicity
    # we always use node 0 for this assignment.

    #identifying root nodes
    
    rn = []
    for w in range(len(factors)):
        if len(factors[w].var) == 1:
            rn.append(factors[w].var[0])
    
    root = rn[0]


    # Create structure to hold messages
    num_nodes = graph.number_of_nodes()
    messages = [0 for _ in range(num_nodes)]
    M = np.array([[None]*num_nodes for _ in range(num_nodes)])
    Mf = [0 for _ in range(num_nodes)]
    m_fac = [1 for _ in range(num_nodes)]


    """ YOUR CODE HERE
    Use the algorithm from lecture 4 and perform message passing over the entire
    graph. Recall the message passing protocol, that a node can only send a
    message to a neighboring node only when it has received messages from all
    its other neighbors.
    Since the provided graphical model is a tree, we can use a two-phase 
    approach. First we send messages inward from leaves towards the root.
    After this is done, we can send messages from the root node outward.
    
    Hint: You might find it useful to add auxilliary functions. You may add 
      them as either inner (nested) or external functions.
    """
    parent = {}
    child = {}
    child1 = {root:[]}
    k_list = list(graph.nodes())
    k_list.remove(root)
    child1.update({k:[] for k in k_list})
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

    #identifying leaf nodes
    leaf_node = [x for (x,v) in child.items() if child[x]==[]]
    mc = []
    for leaf in leaf_node:
        key1 = leaf
        f = Factor()
        f = copy.deepcopy(graph.edges[parent[key1],key1]['factor'])
        if key1 in rn:
            f = factor_product(f, graph.nodes[key1]['factor'])      
        messages[key1] = Factor()
        messages[key1] = factor_marginalize(copy.deepcopy(f), [key1])
        M[key1, parent[key1]] = copy.deepcopy(messages[key1])
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

                joint = copy.deepcopy(messages[ml[0]])

                for i in range(1,len(ml)):
                    joint = factor_product(joint,messages[ml[i]])
                
                if key!=root:
                    
                    joint = factor_product(joint, graph.edges[parent[key],key]['factor'])
                    if key in rn:
                        joint = factor_product(joint, graph.nodes[key]['factor'])
                    joint = factor_marginalize(joint, [key])
                    M[key, parent[key]] = copy.deepcopy(joint)
                    messages[key] = copy.deepcopy(joint)

                if key == root:
                    joint = factor_product(joint, graph.nodes[key]['factor'])
                    messages[key] = copy.deepcopy(joint)
                    Mf[key] = copy.deepcopy(joint)
                    Mf[key].val = np.around(Mf[key].val/np.sum(Mf[key].val),8)
                    m_fac[key] = copy.deepcopy(graph.nodes[root]['factor'])
                    for v1 in value:
                        indi1 = ml[:]
                        indi1.remove(v1)
                        joint1 = copy.deepcopy(m_fac[key])
                        for j1 in indi1:
                            joint1 = factor_product(joint1, M[j1, key])
                        joint1 = factor_product(joint1, graph.edges[v1, key]['factor'])
                        joint1 = factor_marginalize(joint1, [key])
                        M[key,v1] = copy.deepcopy(joint1)
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
                if key in rn:
                    joint2 = copy.deepcopy(graph.nodes[key]['factor'])
                    joint2 = factor_product(joint2, M[parent[key],key])
                else:
                    joint2 = copy.deepcopy(M[parent[key],key])
                indi1 = mld[:]
                indi1.remove(v2)
                for j1 in indi1:
                    joint2 = factor_product(joint2, M[j1, key])
                joint2 = factor_product(joint2, graph.edges[v2, key]['factor'])
                joint2 = factor_marginalize(joint2, [key])
                M[key,v2] = copy.deepcopy(joint2)
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
            joint3.val = np.around(joint3.val/np.sum(joint3.val),8)
            Mf[i] = copy.deepcopy(joint3)
            joint3 = Factor()
                    
    fV = np.array(Mf)[V]
    marginals = list(fV)

    return marginals


def map_eliminate(factors, evidence):
    """Obtains the maximum a posteriori configuration for a tree graph
    given optional evidence

    Args:
        factors (List[Factor]): List of factors representing the graphical model
        evidence (Dict): Observed evidence. evidence[k]=v denotes that the
          variable k is assigned to value v.

    Returns:
        max_decoding (Dict): MAP configuration
        log_prob_max: Log probability of MAP configuration. Note that this is
          log p(MAP, e) instead of p(MAP|e), i.e. it is the unnormalized
          representation of the conditional probability.
    """

    max_decoding = {}
    log_prob_max = 0.0

    """ YOUR CODE HERE
    Use the algorithm from lecture 5 and perform message passing over the entire
    graph to obtain the MAP configuration. Again, recall the message passing 
    protocol.
    Your code should be similar to compute_marginals_bp().
    To avoid underflow, first transform the factors in the probabilities
    to **log scale** and perform all operations on log scale instead.
    You may ignore the warning for taking log of zero, that is the desired
    behavior.
    """
    factors = observe_evidence(factors, evidence)
    graph = generate_graph_from_factors(factors)
    
    
    rn = []
    for w in range(len(factors)):
        if len(factors[w].var) == 1:
            rn.append(factors[w].var[0])
    
    root = rn[0]

    #Log Prob
    for i in range(len(factors)):
        factors[i].val = np.log(factors[i].val)





    num_nodes = graph.number_of_nodes()
    messages = [0 for _ in range(num_nodes)]

    parent = {}
    child = {}
    child1 = {root:[]}
    k_list = list(graph.nodes())
    k_list.remove(root)
    child1.update({k:[] for k in k_list})

    #identifying children of nodes
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
    count = 0

    #similar logic as in compute_bp, first calculate for leaf nodes and start the upward pass
    leaf_node = [x for (x,v) in child.items() if v==[]]
    mc = []
    for leaf in leaf_node:

        key1 = leaf
        f = Factor()
        f = copy.deepcopy(graph.edges[parent[key1],key1]['factor']) 
        if leaf in rn:
            f = factor_sum(f, graph.nodes[leaf]['factor'])       
        messages[key1] = Factor()
        messages[key1] = factor_max_marginalize(copy.deepcopy(f), [key1])
        arg1 = np.argmax(messages[key1].val)
        mc.append(key1)
        count=count+1
        f = None
    
    #Upward pass
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

                joint = copy.deepcopy(messages[ml[0]])

                for i in range(1,len(ml)):
                    joint = factor_sum(joint,messages[ml[i]])
                
                if key!=root:
                
                    joint = factor_sum(joint, graph.edges[parent[key],key]['factor'])
                    if key in rn:
                        joint = factor_sum(joint, graph.nodes[key]['factor'])
                    joint = factor_max_marginalize(joint, [key])
                    messages[key] = copy.deepcopy(joint)
                    arg1 = np.argmax(messages[key].val)

                if key == root:
                    joint = factor_sum(joint, graph.nodes[key]['factor'])
                    messages[key] = copy.deepcopy(joint)
                    arg1 = np.argmax(messages[key].val)
                    max_decoding[key] = arg1
                    log_prob_max = max(messages[key].val)

                joint = None

                mc.append(key)
                count = count+1
    #Implementing Max Decoding
    MD = []
    count = 0
    while count < num_nodes:
        for (key,value) in child.items():
            if key in MD:
                continue
            md2 = child[key]
            if key not in max_decoding.keys():
                continue
            for v2 in md2:
                max_decoding[v2] = messages[v2].val_argmax[max_decoding[key]][v2]
                pass
            MD.append(key)
            count = count + 1    


    #Deleting keys which are in evidence
    for (key,value) in evidence.items():
        del max_decoding[key]
    

    return max_decoding, log_prob_max
