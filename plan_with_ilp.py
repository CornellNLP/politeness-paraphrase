import numpy as np
import json

from perception_utils import estimate_perception

from pulp import LpProblem, LpVariable
from pulp import LpMinimize, LpStatus, lpSum, solvers 

from settings import STRATEGIES, UPPER_BOUND

# strategy_set: set of strategies currently used 
# sender/receiver model: perception model to be used to estimate politeness 
# at_risk_strategies: strategies that are unlikely to pass through the channel 
# strategies: collection of strategies being considered 
# upper_bound: maximum number of strategies to add 

def get_ilp_solution(idx, strategy_set, sender_model, receiver_model, at_risk_strategies, \
                     strategies=STRATEGIES, upper_bound = UPPER_BOUND, intended_politeness= None):
    
    # sender and receiver info
    sender_coefs = sender_model['coefs']
    receiver_coefs = receiver_model['coefs']
    
    sender_intercept = sender_model['intercept']
    receiver_intercept = receiver_model['intercept']
    
    # intention from the sender, perception from the receiver
    if intended_politeness is None:
        intended_val = estimate_perception(sender_model, strategy_set)
    else:
        intended_val = intended_politeness
    
    # create strategy variables
    strategy_vars = LpVariable.dicts("strategies", strategies, \
                                     lowBound=0, upBound=1, cat='Binary')
    
    # objective is to minimize the absolute difference of two predicted politeness scores
    # this is a continuous variable 
    obj_var = LpVariable("y")
    ps_min = LpProblem(idx, LpMinimize)
    ps_min += obj_var
    
    # bound y (i.e., setting up the politeness score objective) 
    ps_min += lpSum([receiver_coefs[name] * strategy_vars[name] for name in strategies]) + receiver_intercept - intended_val <= obj_var
    
    ps_min += intended_val - receiver_intercept - lpSum([receiver_coefs[name] * strategy_vars[name] for name in strategies]) <= obj_var
    
    
    # adding upper bound for new strategies to introduce
    unused_strategies = set(strategies) - set(strategy_set)
    ps_min += lpSum([strategy_vars[name] for name in unused_strategies]) <= upper_bound
    
    
    # add constraint for at risk strategies
    for name in at_risk_strategies:        
        ps_min += strategy_vars[name] == 0
    
    
    # subjunctive/indicative should not be dropped 
    # note: since we are only considering binary variables (i.e., not counts), solutions may not exist (if both strategies are unsafe)
    sub_ind_pair = {"Subjunctive", "Indicative"}
    sub_ind_count = len(sub_ind_pair.intersection(strategy_set))
    ps_min += lpSum([strategy_vars[name] for name in sub_ind_pair]) == sub_ind_count

    # adding negativity constraints
    neg_feat = {k for k,v in receiver_coefs.items() if v < 0}
    if intended_val > 0:
        for name in neg_feat:
            ps_min += strategy_vars[name] == 0
            
    # specify solver (this can be changed/substituted by other options) 
    solver = solvers.GLPK()
    status = ps_min.solve(solver)
    
    # ensure optimal solutin is found
    assert ps_min.status == 1
    
    solution = {name: var.varValue for name, var in strategy_vars.items()}
    solution_set = {k for k,v in solution.items() if v == 1}
    
    return solution_set 
