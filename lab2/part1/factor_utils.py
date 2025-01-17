# taken from part 1
import copy
import numpy as np
from factor import Factor, index_to_assignment, assignment_to_index


def factor_product(A, B):
    """
    Computes the factor product of A and B e.g. A = f(x1, x2); B = f(x1, x3); out=f(x1, x2, x3) = f(x1, x2)f(x1, x3)

    Args:
        A: first Factor
        B: second Factor

    Returns:
        Returns the factor product of A and B
    """
    out = Factor()

    """ YOUR CODE HERE """
    if A.is_empty():
        return B
    if B.is_empty():
        return A

    # Create output factor. Variables should be the union between of the
    # variables contained in the two input factors
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
    out.val = np.multiply(A.val[idxA],B.val[idxB])

    """ END YOUR CODE HERE """
    return out


def factor_marginalize(factor_1, var):
    """
    Returns factor after variables in var have been marginalized out.

    Args:
        factor: factor to be marginalized
        var: numpy array of variables to be marginalized over

    Returns:
        marginalized factor
    """
    if len(var) == 0:
        return factor_1

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

    """ END YOUR CODE HERE """
    return out


def factor_evidence(factor, evidence):
    """
    Observes evidence and retains entries containing the observed evidence. Also removes the evidence random variables
    because they are already observed e.g. factor=f(1, 2) and evidence={1: 0} returns f(2) with entries from node1=0
    Args:
        factor: factor to reduce using evidence
        evidence:  dictionary of node:evidence pair where evidence[1] = evidence of node 1.
    Returns:
        Reduced factor that does not contain any variables in the evidence. Return an empty factor if all the
        factor's variables are observed.
    """
    out = Factor()

    """ YOUR CODE HERE,     HINT: copy from lab2 part 1! """

    if len(evidence)==0:
        return factor

    out.var = np.array(list(set(factor.var) - set(list(evidence.keys()))))

    keep_list = []
    del_list = []

    for (key,value) in evidence.items():
        k1 = key
        v1 = value
        factor_var = factor.var
        if k1 in factor_var:
            fa = factor.get_all_assignments()
            i1 = np.where(factor_var == k1)
            del_list.append(i1)
            i2 = np.where(fa[:,i1[0][0]] == v1)
            keep_list.extend(i2[0])
        
    out.val = factor.val[keep_list]
    out.card = np.delete(factor.card,del_list, axis=0)
    ass = fa[keep_list]

    dce = []
    ek = evidence.keys()
    for i in ek:
        dc = np.where(i == factor.var)[0]
        dce.append(dc[0])
    ds = list(set(np.arange(0,len(factor.var),1)) - set(dce))
    oa = out.get_all_assignments()
    cov = [0.0] * len(out.val)
    for i in range(len(oa)):
        i1 = np.where(ass[:,ds][i] == oa)[0][0]
        cov[i1] = out.val[i]
    
    cov = np.array(cov)
    out.val = cov[:]



    """ END YOUR CODE HERE """


    return out



# if __name__ == '__main__':
#     main()
