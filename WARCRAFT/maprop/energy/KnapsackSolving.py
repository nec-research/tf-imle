import time
#from ortools.algorithms import pywrapknapsack_solver
import numpy as np
from scipy import stats
from sklearn.metrics.scorer import _BaseScorer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from gurobipy import *
def solveKnapsackGreedily(profits, weights, capacity):
    assert len(profits) == len(weights)
    assert(len([x for x in profits if x < 0]) == 0)
    assert(capacity > 0 and len(profits) >= 2)    
    
    n_items = len(profits)    
    items = [[profits[i], weights[i], i] for i in range(n_items) ]
    items.sort(key=lambda x: float(x[0])/x[1], reverse=True)
    assert(items[0][0]/items[0][1] >= items[1][0]/items[1][1])
    
    objective = 0
    available_capacity = capacity
    assignments = [0 for i in range(n_items)] #the ith value is 1 if the i-th item is selected
    for i in range(n_items):
        #if the item cannot fit in the remaining capacity
        if items[i][1] > available_capacity:
            continue
        #else, insert it into the knapsack
        objective += items[i][0] #profit increase
        available_capacity -= items[i][1] #weight
        assignments[items[i][2]] = 1 #note that you selected this item; items[i][2] indicates the original index of the item
    
        if available_capacity == 0:
            break
        
    solution_info = {'objective':objective, 'assignments':assignments}
    return solution_info

#returns a dictionary d with the following info about the solution:
#   d['objective'] -> the objective value
#   d['assignments'] -> an array of 0-1 values, indicating the assignments for each item (if it is included or not)
#   d['runtime'] -> time in seconds
# TIME_LIMIT is ignored!!!
'''
def solveKnapsackProblem(profits, weights, capacity, time_limit=None, use_dp=True,multi= False):
    #assert len(profits) == len(weights)
    #print(profits)
    #assert(len([x for x in profits if x < 0]) == 0)
    #if len([x for x in profits if x < 0]) != 0:
        #print("WARNING: NEGATIVE PROFITS IN KNAPSACK SOLVING!\n\n!!!!!!!!!!\n\n")
        #print(profits)
        #print(weights)
    multi = (len(weights)>1)
    if use_dp:
        if multi:
            solver_type = pywrapknapsack_solver.KnapsackSolver.KNAPSACK_MULTIDIMENSION_CBC_MIP_SOLVER
        else:
            solver_type = pywrapknapsack_solver.KnapsackSolver.KNAPSACK_DYNAMIC_PROGRAMMING_SOLVER
    else:
        solver_type = pywrapknapsack_solver.KnapsackSolver.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER
    solver = pywrapknapsack_solver.KnapsackSolver(solver_type, "ortools_kn")
    
    cst = 1e+8
    profits_lng = [int(v*cst) for v in profits]
    weights = [[int(w) for w in W] for W in weights]
    capacity = [int(c) for c in capacity]
    start_time = time.time()
    
    try:
        solver.Init(profits_lng, weights, capacity)
        val = solver.Solve()/cst
    
        solution_info = {}
        solution_info['runtime'] = time.time() - start_time
        solution_info['objective'] = val
        solution_info['assignments'] = [int(solver.BestSolutionContains(x)) for x in range(len(profits))]
        
    except:
        print("SOME EXCEPTION HAPPENED! RETURNING GARBAGE\n")
        val = 0
        import sys
            
        solution_info = {}
        solution_info['runtime'] = time.time() - start_time
        solution_info['objective'] = val
        solution_info['assignments'] = [0 for x in range(len(profits))]
        

    return solution_info

# returns the TRUE values of the knapsacks obtained by y_pred
'''
def solveKnapsackProblemRelaxation(profits, weights, capacity, warmstart=None,time_limit=None, use_dp=True):
    #assert len(profits) == len(weights)
    #print(profits)
    #assert(len([x for x in profits if x < 0]) == 0)
    #if len([x for x in profits if x < 0]) != 0:
        #print("WARNING: NEGATIVE PROFITS IN KNAPSACK SOLVING!\n\n!!!!!!!!!!\n\n")
        #print(profits)
        #print(weights)
    multi = (len(weights)>1)
    
    profits = [v for v in profits]
    weights = [[w for w in W] for W in weights]
    capacity = [c for c in capacity]
    start_time = time.time()
    n = len(profits)
    
    
    m = Model()
    m.setParam('OutputFlag', 0)
    x = {}
    for i in range(n):
        x[(i)] = m.addVar(lb=0,ub=1, name= "x"+str(i))
        #x[i] = m.addVar(vtype=GRB.BINARY,name= "x"+str(i))
    
    if warmstart is not None:
        for i in range(n):
            x[i].Pstart = warmstart[i]
            m.update()
 
    m.setObjective(sum( (x[i]*profits[i]) for i in range(n)), GRB.MAXIMIZE)
    for w in weights:
        for c in capacity:    
            m.addConstr(( quicksum(x[i]*w[i] for i in range(n) ) <= c))
    '''    
    lb= [0.0]*n
    ub = [1.0]*n
    x = m.addVars(n,ub=ub, name='x')
    for w in weights:
        for c in capacity:
            m.addConstr(x.prod(w) <= c)
    m.setObjective(x.prod(profits), GRB.MAXIMIZE)
    '''
    m.optimize()       

    solution_info = {}
    if (m.status == GRB.Status.OPTIMAL):
        solution_info['runtime'] = m.Runtime
        solution_info['objective'] = m.objVal
        m_on = m.getAttr('x',x)
        sol = list(m_on.values())     
        solution_info['assignments'] =  [i for i in sol]
    else:
        print("SOME EXCEPTION HAPPENED! RETURNING GARBAGE\n")
        val = 0
        import sys
            
        solution_info = {}
        solution_info['runtime'] = m.Runtime
        solution_info['objective'] = val
        solution_info['assignments'] = [0 for x in range(len(profits))]
    return solution_info

def solveKnapsackProblem(profits, weights, capacity,warmstart=None, time_limit=None, use_dp=True):
    #assert len(profits) == len(weights)
    #print(profits)
    #assert(len([x for x in profits if x < 0]) == 0)
    #if len([x for x in profits if x < 0]) != 0:
        #print("WARNING: NEGATIVE PROFITS IN KNAPSACK SOLVING!\n\n!!!!!!!!!!\n\n")
        #print(profits)
        #print(weights)
    multi = (len(weights)>1)
    
    profits = [v for v in profits]
    weights = [[w for w in W] for W in weights]
    capacity = [c for c in capacity]
    start_time = time.time()
    n = len(profits)
    
    
    m = Model()
    m.setParam('OutputFlag', 0)
    x = {}
    for i in range(n):
        x[(i)] = m.addVar(lb=0,ub=1, name= "x"+str(i))
        #x[i] = m.addVar(vtype=GRB.BINARY,name= "x"+str(i))
    
    if warmstart is not None:
        for i in range(n):
            x[i].Pstart = warmstart[i]
            m.update()
 
    m.setObjective(sum( (x[i]*profits[i]) for i in range(n)), GRB.MAXIMIZE)
    for w in weights:
        for c in capacity:    
            m.addConstr(( quicksum(x[i]*w[i] for i in range(n) ) <= c))
    '''

    x = m.addVars(n,vtype=GRB.BINARY, name='x')
    for w in weights:
        for c in capacity:
            m.addConstr(x.prod(w) <= c)
    m.setObjective(x.prod(profits), GRB.MAXIMIZE)
    '''
    m.optimize()       

    solution_info = {}
    try:
        if (m.status == GRB.Status.OPTIMAL):
            solution_info['runtime'] = m.Runtime
            solution_info['objective'] = m.objVal
            m_on = m.getAttr('x',x)
            sol = list(m_on.values())     
            solution_info['assignments'] =  [int(i) for i in sol]
    except:
        print("SOME EXCEPTION HAPPENED! RETURNING GARBAGE\n")
        val = 0
        import sys
            
        solution_info = {}
        solution_info['runtime'] = m.Runtime
        solution_info['objective'] = val
        solution_info['assignments'] = [0 for x in range(len(profits))]
    return solution_info

def eval_knapsack(grpY_true, grpY_pred, weights, cap, greedy=False,relaxation= False):
    if isinstance(weights, str) and weights == 'uniform':
        weights = np.ones(len(grpY_true[0]))
    assert(isinstance(weights,np.ndarray))
    
    vals = np.zeros(len(grpY_true))
    assignments = []
    for i in range(len(grpY_true)):
        knap_sol = {}
        if greedy:
            from knapsack_solving import solveKnapsackGreedily
            knap_sol = solveKnapsackGreedily(profits=grpY_pred[i], weights=weights, capacity=cap)
        if relaxation:
            knap_sol = solveKnapsackProblemRelaxation(grpY_pred[i], weights, cap)
        else:
            knap_sol = solveKnapsackProblem(grpY_pred[i], weights, cap)
        vals[i] =np.sum(grpY_true[i] * np.array(knap_sol['assignments'])) 
        assignments.append(knap_sol['assignments'])

    return vals,assignments

def regret_knapsack(grpY_true, grpY_pred, weights='uniform', cap=10,assignments_true=None, relaxation=False):
    # if called repeatedly, vals_true should be cached
    if assignments_true is None:
        vals_true,assignments_true = eval_knapsack(grpY_true, grpY_true, weights=weights, cap=cap, relaxation= relaxation)
    else:
        vals_true =  np.sum(grpY_true* np.array(assignments_true)) 
    vals_pred, assignments_pred = eval_knapsack(grpY_true, grpY_pred, weights=weights, cap=cap, relaxation= relaxation)
    from sklearn.metrics import confusion_matrix
    if relaxation:
        confusion_mat = np.zeros((2,2))
    else:
        confusion_mat = confusion_matrix(assignments_true[0], assignments_pred[0],labels=[0,1])

    return np.average(vals_true - vals_pred),confusion_mat
def knapsack_diversity(y_test, capacity,weights,n_items =48):
    def entropy(p):
        return -(p*np.log10(p)+(1-p)*np.log10(1-p))
    def get_profits(trch_y, kn_nr, n_items):
        kn_start = kn_nr*n_items
        kn_stop = kn_start+n_items
        return trch_y[kn_start:kn_stop].data.numpy().T[0]
    n_knap = y_test.shape[0]//n_items
    import torch
    trch_y_test = torch.from_numpy(np.array([y_test]).T).float()
    entropy_list = np.zeros(n_knap)
    ones_assigned =0
    zeros_assigned =0
    for kn_nr in range(n_knap):
        V_true = get_profits(trch_y_test, kn_nr, n_items)
        knap_sol = solveKnapsackProblem(V_true,weights=weights,capacity=capacity)
        ones_assigned += np.sum(np.array(knap_sol['assignments']))
        zeros_assigned += n_items - np.sum(np.array(knap_sol['assignments']))
    percent_one = ones_assigned/y_test.shape[0]
    percent_zero =zeros_assigned/y_test.shape[0]
    return entropy(percent_zero),min(percent_zero,percent_one)




