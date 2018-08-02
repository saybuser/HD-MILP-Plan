import math

import cplex
from cplex.exceptions import CplexError

def sparsifyDNN(sparsificationThreshold, weights, bias, inputNeurons, mappings, relus, outputs):
    
    orderedAbsWeights = sorted(map(abs, list(weights.values())+list(bias.values())))
    index = int(math.floor(sparsificationThreshold*(len(orderedAbsWeights)-1)))
    cutoffWeight = orderedAbsWeights[index]
    
    for relu in relus:
        for inp in inputNeurons[(relu)]:
            if inp not in mappings:
                if abs(weights[(inp,relu)]) <= cutoffWeight:
                    weights[(inp,relu)] = 0.0
        if abs(bias[(relu)]) <= cutoffWeight:
            bias[(relu)] = 0.0

    for output in outputs:
        for inp in inputNeurons[(output)]:
            if inp not in mappings:
                if abs(weights[(inp,output)]) <= cutoffWeight:
                    weights[(inp,output)] = 0.0
        if abs(bias[(output)]) <= cutoffWeight:
            bias[(output)] = 0.0

    return weights, bias

def readDNN(directory):
    
    inputNeurons = {}
    weights = {}
    bias = {}
    activationType = {}
    
    DNNFile = open(directory,"r")
    lines = DNNFile.read().splitlines()
    
    for line in lines:
        data = line.split(",")
        for index, dat in enumerate(data):
            if index == 0:
                inputNeurons[(dat)] = []
            elif index == 1:
                activationType[(data[0])] = dat
            else:
                if index % 2 == 0:
                    if dat[0] != "B":
                        inputNeurons[(data[0])].append(dat)
                else:
                    if data[index-1][0] == "B":
                        bias[(data[0])] = float(dat)
                    else:
                        weights[(data[index-1],data[0])] = float(dat)

    return inputNeurons, weights, bias, activationType

def readInitial(directory):
    
    initial = []
    initialFile = open(directory,"r")
    data = initialFile.read().splitlines()
    
    for dat in data:
        initial.append(dat.split(","))

    return initial

def readGoal(directory):
    
    import os
    
    goals = []
    
    if os.path.exists(directory):
        goalsFile = open(directory,"r")
        data = goalsFile.read().splitlines()
    
        for dat in data:
            goals.append(dat.split(","))
    else:
        print("No goal file provided.")
    
    return goals

def readMappings(directory):
    
    mappings = {}
    mappingsFile = open(directory,"r")
    data = mappingsFile.read().splitlines()
    
    for dat in data:
        key, value = dat.split(",")
        mappings[key] = value
    
    return mappings

def readConstraints(directory):
    
    import os
    
    constraints = []
    
    if os.path.exists(directory):
        constraintsFile = open(directory,"r")
        data = constraintsFile.read().splitlines()
    
        for dat in data:
            constraints.append(dat.split(","))
    else:
        print("No constraint file provided.")

    return constraints

def readTransitions(directory):
    
    import os
    
    transitions = []
    
    if os.path.exists(directory):
        transitionsFile = open(directory,"r")
        data = transitionsFile.read().splitlines()
    
        for dat in data:
            transitions.append(dat.split(","))
    else:
        print("No known transition file provided.")
    
    return transitions

def readReward(directory):
    
    import os
    
    reward = []
    
    if os.path.exists(directory):
        rewardFile = open(directory,"r")
        data = rewardFile.read().splitlines()
    
        for dat in data:
            reward.append(dat.split(","))
    else:
        print("No reward file provided.")
    
    return reward

def readVariables(directory):
    
    A = []
    S = []
    Aux = []
    A_type = []
    S_type = []
    Aux_type = []
    
    variablesFile = open(directory,"r")
    data = variablesFile.read().splitlines()
    
    for dat in data:
        variables = dat.split(",")
        for var in variables:
            if "action_continuous:" in var or "action_boolean:" in var or "action_integer:" in var:
                if "action_continuous:" in var:
                    A.append(var.replace("action_continuous: ",""))
                    A_type.append("C")
                elif "action_boolean:" in var:
                    A.append(var.replace("action_boolean: ",""))
                    A_type.append("B")
                else:
                    A.append(var.replace("action_integer: ",""))
                    A_type.append("I")
            elif "state_continuous:" in var or "state_boolean:" in var or "state_integer:" in var:
                if "state_continuous:" in var:
                    S.append(var.replace("state_continuous: ",""))
                    S_type.append("C")
                elif "state_boolean:" in var:
                    S.append(var.replace("state_boolean: ",""))
                    S_type.append("B")
                else:
                    S.append(var.replace("state_integer: ",""))
                    S_type.append("I")
            else:
                if "auxiliary_continuous:" in var:
                    Aux.append(var.replace("auxiliary_continuous: ",""))
                    Aux_type.append("C")
                elif "auxiliary_boolean:" in var:
                    Aux.append(var.replace("auxiliary_boolean: ",""))
                    Aux_type.append("B")
                else:
                    Aux.append(var.replace("auxiliary_integer: ",""))
                    Aux_type.append("I")

    return A, S, Aux, A_type, S_type, Aux_type

def initialize_variables(c, A, S, Aux, relus, A_type, S_type, Aux_type, horizon):
    
    VARINDEX = 0
    
    vartypes = ""
    colnames = []
    
    # Create vars for each action a, time step t
    x = {}
    for index, a in enumerate(A):
        for t in range(horizon):
            x[(a,t)] = VARINDEX
            colnames.append(str(x[(a,t)]))
            vartypes += A_type[index]
            VARINDEX += 1

    # Create vars for each state s, time step t
    y = {}
    for index, s in enumerate(S):
        for t in range(horizon+1):
            y[(s,t)] = VARINDEX
            colnames.append(str(y[(s,t)]))
            vartypes += S_type[index]
            VARINDEX += 1

    # Create vars for each auxilary variable aux, time step t
    v = {}
    for index,aux in enumerate(Aux):
        for t in range(horizon+1):
            v[(aux,t)] = VARINDEX
            colnames.append(str(v[(aux,t)]))
            vartypes += Aux_type[index]
            VARINDEX += 1

    # Create vars for each relu node z, time step t
    z = {}
    zPrime = {}
    for relu in relus:
        for t in range(horizon):
            z[(relu,t)] = VARINDEX
            colnames.append(str(z[(relu,t)]))
            vartypes += "C"
            VARINDEX += 1
            zPrime[(relu,t)] = VARINDEX
            colnames.append(str(zPrime[(relu,t)]))
            vartypes += "B"
            VARINDEX += 1

    lbs = [-1.0*cplex.infinity] * VARINDEX
    ubs = [cplex.infinity] * VARINDEX
    
    c.variables.add(types=vartypes, names=colnames, lb = lbs, ub = ubs)
    
    return c, x, y, v, z, zPrime, vartypes, colnames

def encode_initial_constraints(c, initial, y):
    
    for init in initial:
        variables = init[:-2]
        literals = []
        coefs = []
        for var in variables:
            coef = "1.0"
            if "*" in var:
                coef, var = var.split("*")
            literals.append(y[(var,0)])
            coefs.append(float(coef))
        RHS = float(init[len(init)-1])
        if "<=" == init[len(init)-2]:
            row = [ [ literals, coefs ] ]
            c.linear_constraints.add(lin_expr=row, senses="L", rhs=[RHS])
        elif ">=" == init[len(init)-2]:
            row = [ [ literals, coefs ] ]
            c.linear_constraints.add(lin_expr=row, senses="G", rhs=[RHS])
        else:
            row = [ [ literals, coefs ] ]
            c.linear_constraints.add(lin_expr=row, senses="E", rhs=[RHS])
    
    return c

def encode_goal_constraints(c, goals, y, horizon):
    
    for goal in goals:
        variables = goal[:-2]
        literals = []
        coefs = []
        for var in variables:
            coef = "1.0"
            if "*" in var:
                coef, var = var.split("*")
            literals.append(y[(var,horizon)])
            coefs.append(float(coef))
        RHS = float(goal[len(goal)-1])
        if "<=" == goal[len(goal)-2]:
            row = [ [ literals, coefs ] ]
            c.linear_constraints.add(lin_expr=row, senses="L", rhs=[RHS])
        elif ">=" == goal[len(goal)-2]:
            row = [ [ literals, coefs ] ]
            c.linear_constraints.add(lin_expr=row, senses="G", rhs=[RHS])
        else:
            row = [ [ literals, coefs ] ]
            c.linear_constraints.add(lin_expr=row, senses="E", rhs=[RHS])

    return c

def encode_global_constraints(c, constraints, A, S, Aux, x, y, v, horizon):
    
    for t in range(horizon+1):
        for constraint in constraints:
            variables = constraint[:-2]
            literals = []
            coefs = []
            if set(A).isdisjoint(variables) or t < horizon: # for the last time step, only consider constraints that include states variables-only
                for var in variables:
                    coef = "1.0"
                    if "*" in var:
                        coef, var = var.split("*")
                    if var in A:
                        literals.append(x[(var,t)])
                        coefs.append(float(coef))
                    elif var in S:
                        literals.append(y[(var,t)])
                        coefs.append(float(coef))
                    else:
                        literals.append(v[(var,t)])
                        coefs.append(float(coef))
                RHS = float(constraint[len(constraint)-1])
                if "<=" == constraint[len(constraint)-2]:
                    row = [ [ literals, coefs ] ]
                    c.linear_constraints.add(lin_expr=row, senses="L", rhs=[RHS])
                elif ">=" == constraint[len(constraint)-2]:
                    row = [ [ literals, coefs ] ]
                    c.linear_constraints.add(lin_expr=row, senses="G", rhs=[RHS])
                else:
                    row = [ [ literals, coefs ] ]
                    c.linear_constraints.add(lin_expr=row, senses="E", rhs=[RHS])
    
    return c

def encode_known_transitions(c, transitions, A, S, Aux, x, y, v, horizon):
    
    for t in range(horizon):
        for transition in transitions:
            variables = transition[:-2]
            literals = []
            coefs = []
            for var in variables:
                coef = "1.0"
                if "*" in var:
                    coef, var = var.split("*")
                if var in A:
                    literals.append(x[(var,t)])
                    coefs.append(float(coef))
                elif var in Aux:
                    literals.append(v[(var,t)])
                    coefs.append(float(coef))
                else:
                    if var[len(var)-1] == "'":
                        literals.append(y[(var[:-1],t+1)])
                    else:
                        literals.append(y[(var,t)])
                    coefs.append(float(coef))
            RHS = float(transition[len(transition)-1])
            if "<=" == transition[len(transition)-2]:
                row = [ [ literals, coefs ] ]
                c.linear_constraints.add(lin_expr=row, senses="L", rhs=[RHS])
            elif ">=" == transition[len(transition)-2]:
                row = [ [ literals, coefs ] ]
                c.linear_constraints.add(lin_expr=row, senses="G", rhs=[RHS])
            else:
                row = [ [ literals, coefs ] ]
                c.linear_constraints.add(lin_expr=row, senses="E", rhs=[RHS])

    return c

def encode_activation_constraints(c, relus, bias, inputNeurons, mappings, weights, A, S, x, y, z, zPrime, bigM, horizon):
    
    for t in range(horizon):
        for relu in relus:
            
            row = [ [ [z[(relu,t)]], [1.0] ] ]
            c.linear_constraints.add(lin_expr=row, senses="G", rhs=[0.0])
            
            row = [ [ [z[(relu,t)], zPrime[(relu,t)]], [1.0, -1.0*bigM] ] ]
            c.linear_constraints.add(lin_expr=row, senses="L", rhs=[0.0])
            
            inputs = []
            coefs = []
            RHS = -1.0*bias[(relu)]
            for inp in inputNeurons[(relu)]:
                if inp in mappings:
                    coefs.append(weights[(inp,relu)])
                    if mappings[(inp)] in A:
                        inputs.append(x[(mappings[(inp)],t)])
                    else:
                        inputs.append(y[(mappings[(inp)],t)])
                else:
                    coefs.append(weights[(inp,relu)])
                    inputs.append(z[(inp,t)])
        
            row = [ [ inputs + [z[(relu,t)]], coefs + [-1.0] ] ]
            c.linear_constraints.add(lin_expr=row, senses="L", rhs=[RHS])
            
            RHS += -1.0*bigM
            row = [ [ inputs + [z[(relu,t)]] + [zPrime[(relu,t)]], coefs + [-1.0] + [-1.0*bigM] ] ]
            c.linear_constraints.add(lin_expr=row, senses="G", rhs=[RHS])
    
    return c

def encode_nextstate_constraints(c, outputs, bias, inputNeurons, mappings, weights, A, S, x, y, z, activationType, S_type, bigM, horizon):
    
    for t in range(1,horizon+1):
        for output in outputs:
            inputs = []
            coefs = []
            RHS = -1.0*bias[(output)]
            for inp in inputNeurons[(output)]:
                if inp in mappings:
                    coefs.append(weights[(inp,output)])
                    if mappings[(inp)] in A:
                        inputs.append(x[(mappings[(inp)],t-1)])
                    else:
                        inputs.append(y[(mappings[(inp)],t-1)])
                else:
                    coefs.append(weights[(inp,output)])
                    inputs.append(z[(inp,t-1)])
        
            if activationType[(output)] == "linear" and S_type[S.index(mappings[(output)])] == "C":
                row = [ [ inputs + [y[(mappings[(output)],t)]], coefs + [-1.0] ] ]
                c.linear_constraints.add(lin_expr=row, senses="E", rhs=[RHS])
            elif activationType[(output)] == "linear" and S_type[S.index(mappings[(output)])] == "I":
                row = [ [ inputs + [y[(mappings[(output)],t)]], coefs + [-1.0] ] ]
                c.linear_constraints.add(lin_expr=row, senses="L", rhs=[RHS + 0.5])
                row = [ [ inputs + [y[(mappings[(output)],t)]], coefs + [-1.0] ] ]
                c.linear_constraints.add(lin_expr=row, senses="G", rhs=[-1.0*RHS - 0.5])
            elif activationType[(output)] == "step" and S_type[S.index(mappings[(output)])] == "B":
                row = [ [ inputs + [y[(mappings[(output)],t)]], coefs + [-1.0*bigM] ] ]
                c.linear_constraints.add(lin_expr=row, senses="L", rhs=[RHS])
                row = [ [ inputs + [y[(mappings[(output)],t)]], coefs + [-1.0*bigM] ] ]
                c.linear_constraints.add(lin_expr=row, senses="G", rhs=[RHS - bigM])
            else:
                print ("This activation function/state domain combination is currently not supported.")

    return c

def encode_reward(c, reward, colnames, A, S, Aux, x, y, v, horizon):
    
    objcoefs = [0.0]*len(colnames)
    
    for t in range(horizon):
        for var, weight in reward:
            if var in A or var[1:] in A:
                objcoefs[colnames.index(str(x[(var,t)]))] = -1.0*float(weight)
            elif var in S or var[1:] in S:
                objcoefs[colnames.index(str(y[(var,t+1)]))] = -1.0*float(weight)
            else:
                objcoefs[colnames.index(str(v[(var,t+1)]))] = -1.0*float(weight)

    for index, obj in enumerate(objcoefs):
        c.objective.set_linear([(index, obj)])
    
    return c

def initialize_strengthening_variables(c, colnames, A, S, A_lb, A_ub, S_lb, S_ub, horizon):
    
    VARINDEX = len(colnames)
    
    vartypes = ""
    colnames = []
    objcoefs = []
    lbs = []
    ubs = []
    
    # Create vars for each action a, time step t
    x_plus = {}
    x_minus = {}
    xPrime = {}
    for a in A:
        for t in range(horizon):
            if A_lb[(a,t)] < 0.0 and A_ub[(a,t)] > 0.0:
                x_plus[(a,t)] = VARINDEX
                colnames.append(str(x_plus[(a,t)]))
                objcoefs.append(0.0)
                lbs.append(0.0)
                ubs.append(A_ub[(a,t)])
                vartypes += "C"
                VARINDEX += 1
                x_minus[(a,t)] = VARINDEX
                colnames.append(str(x_minus[(a,t)]))
                objcoefs.append(0.0)
                lbs.append(A_lb[(a,t)])
                ubs.append(0.0)
                vartypes += "C"
                VARINDEX += 1
                xPrime[(a,t)] = VARINDEX
                colnames.append(str(xPrime[(a,t)]))
                objcoefs.append(0.0)
                lbs.append(0.0)
                ubs.append(1.0)
                vartypes += "B"
                VARINDEX += 1

    # Create vars for each state s, time step t
    y_plus = {}
    y_minus = {}
    yPrime = {}
    for index, s in enumerate(S):
        for t in range(horizon+1):
            if S_lb[(s,t)] < 0.0 and S_ub[(s,t)] > 0.0:
                y_plus[(s,t)] = VARINDEX
                colnames.append(str(y_plus[(s,t)]))
                objcoefs.append(0.0)
                lbs.append(0.0)
                ubs.append(S_ub[(s,t)])
                vartypes += "C"
                VARINDEX += 1
                y_minus[(s,t)] = VARINDEX
                colnames.append(str(y_plus[(s,t)]))
                objcoefs.append(0.0)
                lbs.append(S_lb[(s,t)])
                ubs.append(0.0)
                vartypes += "C"
                VARINDEX += 1
                yPrime[(s,t)] = VARINDEX
                colnames.append(str(yPrime[(s,t)]))
                objcoefs.append(0.0)
                lbs.append(0.0)
                ubs.append(1.0)
                vartypes += "B"
                VARINDEX += 1

    c.variables.add(types=vartypes, names=colnames, lb = lbs, ub = ubs)
    
    return c, x_plus, x_minus, xPrime, y_plus, y_minus, yPrime

def encode_improvedbound_constraints(c, A, S, colnames, x, y, horizon):
    
    A_lb = {}
    A_ub = {}
    S_lb = {}
    S_ub = {}
    
    print("Preprocessing bounds.")
    
    c.set_log_stream(None)
    c.set_error_stream(None)
    c.set_warning_stream(None)
    c.set_results_stream(None)
    
    # Set search emphasis to improving bounds
    c.parameters.emphasis.mip.set(3)
    
    # Total deterministic time allocated to preprocessing
    totaltime = 60000.0
    
    # Allocate time to each action per time
    timepervar = (totaltime/10.0)/float(horizon*len(A))
    
    # Set deterministic time limit
    c.parameters.dettimelimit.set(timepervar)
    
    # Perform reachability on state and action variables to obtain tighter bounds
    for t in range(horizon):
        for a in A:
            objcoefs = [0.0]*len(colnames)
            objcoefs[colnames.index(str(x[(a,t)]))] = 1.0
            for index, obj in enumerate(objcoefs):
                c.objective.set_linear([(index, obj)])
            c.solve()
            A_lb[(a,t)] = c.solution.MIP.get_best_objective()
            c.variables.set_lower_bounds([(colnames.index(str(x[(a,t)])), c.solution.MIP.get_best_objective())])
            #row = [ [ [x[(a,t)]], [1.0] ] ]
            #c.linear_constraints.add(lin_expr=row, senses="G", rhs=[c.solution.MIP.get_best_objective()])
            
            objcoefs = [0.0]*len(colnames)
            objcoefs[colnames.index(str(x[(a,t)]))] = -1.0
            for index, obj in enumerate(objcoefs):
                c.objective.set_linear([(index, obj)])
            c.solve()
            A_ub[(a,t)] = -1.0*c.solution.MIP.get_best_objective()
            c.variables.set_upper_bounds([(colnames.index(str(x[(a,t)])), -1.0*c.solution.MIP.get_best_objective())])
            #row = [ [ [x[(a,t)]], [1.0] ] ]
            #c.linear_constraints.add(lin_expr=row, senses="L", rhs=[-1.0*c.solution.MIP.get_best_objective()])

    # Total time left allocated to preprocessing
    totaltime -= timepervar*float(horizon*len(A))
    
    # Allocate time to each state per time
    timepervar = totaltime/float((horizon+1)*len(S))

    # Set deterministic time limit
    c.parameters.dettimelimit.set(timepervar)

    for t in range(horizon+1):
        for s in S:
            objcoefs = [0.0]*len(colnames)
            objcoefs[colnames.index(str(y[(s,t)]))] = 1.0
            for index, obj in enumerate(objcoefs):
                c.objective.set_linear([(index, obj)])
            c.solve()
            S_lb[(s,t)] = c.solution.MIP.get_best_objective()
            c.variables.set_lower_bounds([(colnames.index(str(y[(s,t)])), c.solution.MIP.get_best_objective())])
            #row = [ [ [y[(s,t)]], [1.0] ] ]
            #c.linear_constraints.add(lin_expr=row, senses="G", rhs=[c.solution.MIP.get_best_objective()])
            
            objcoefs = [0.0]*len(colnames)
            objcoefs[colnames.index(str(y[(s,t)]))] = -1.0
            for index, obj in enumerate(objcoefs):
                c.objective.set_linear([(index, obj)])
            c.solve()
            S_ub[(s,t)] = -1.0*c.solution.MIP.get_best_objective()
            c.variables.set_upper_bounds([(colnames.index(str(y[(s,t)])), -1.0*c.solution.MIP.get_best_objective())])
            #row = [ [ [y[(s,t)]], [1.0] ] ]
            #c.linear_constraints.add(lin_expr=row, senses="L", rhs=[-1.0*c.solution.MIP.get_best_objective()])

    # Reset search emphasis to default
    c.parameters.emphasis.mip.reset()

    # Reset deterministic time limit
    c.parameters.dettimelimit.reset()

    # Reset optimizer log settings
    import sys
    
    c.set_log_stream(sys.stdout)
    c.set_error_stream(sys.stderr)
    c.set_warning_stream(sys.stderr)
    c.set_results_stream(sys.stdout)

    return c, A_lb, A_ub, S_lb, S_ub

def encode_strengthened_activation_constraints(c, A, S, relus, bias, inputNeurons, mappings, weights, colnames, x, y, z, zPrime, horizon):
    
    #Set tighter bound constraints
    c, A_lb, A_ub, S_lb, S_ub = encode_improvedbound_constraints(c, A, S, colnames, x, y, horizon)
    
    #Initialize variables for strengthening constraints
    c, x_plus, x_minus, xPrime, y_plus, y_minus, yPrime = initialize_strengthening_variables(c, colnames, A, S, A_lb, A_ub, S_lb, S_ub, horizon)
    
    #Add auxillary constraints to relate auxiliary variables to action and state variables
    for a in A:
        for t in range(horizon):
            if A_lb[(a,t)] < 0.0 and A_ub[(a,t)] > 0.0:
                row = [ [ [x[(a,t)]] + [x_plus[(a,t)]] + [x_minus[(a,t)]], [1.0] + [-1.0] + [-1.0] ] ]
                c.linear_constraints.add(lin_expr=row, senses="E", rhs=[0.0])

                row = [ [ [x[(a,t)]] + [xPrime[(a,t)]], [1.0] + [-1.0*A_ub[(a,t)]] ] ]
                c.linear_constraints.add(lin_expr=row, senses="L", rhs=[0.0])

                row = [ [ [x[(a,t)]] + [xPrime[(a,t)]], [1.0] + [A_lb[(a,t)]] ] ]
                c.linear_constraints.add(lin_expr=row, senses="G", rhs=[A_lb[(a,t)]])

                row = [ [ [x_plus[(a,t)]] + [xPrime[(a,t)]], [1.0] + [-1.0*A_ub[(a,t)]] ] ]
                c.linear_constraints.add(lin_expr=row, senses="L", rhs=[0.0])

                row = [ [ [x_minus[(a,t)]] + [xPrime[(a,t)]], [1.0] + [A_lb[(a,t)]] ] ]
                c.linear_constraints.add(lin_expr=row, senses="G", rhs=[A_lb[(a,t)]])

    for s in S:
        for t in range(horizon+1):
            if S_lb[(s,t)] < 0.0 and S_ub[(s,t)] > 0.0:
                row = [ [ [y[(s,t)]] + [y_plus[(s,t)]] + [y_minus[(s,t)]], [1.0] + [-1.0] + [-1.0] ] ]
                c.linear_constraints.add(lin_expr=row, senses="E", rhs=[0.0])

                row = [ [ [y[(s,t)]] + [yPrime[(s,t)]], [1.0] + [-1.0*S_ub[(s,t)]] ] ]
                c.linear_constraints.add(lin_expr=row, senses="L", rhs=[0.0])
        
                row = [ [ [y[(s,t)]] + [yPrime[(s,t)]], [1.0] + [S_lb[(s,t)]] ] ]
                c.linear_constraints.add(lin_expr=row, senses="G", rhs=[S_lb[(s,t)]])
                
                row = [ [ [y_plus[(s,t)]] + [yPrime[(s,t)]], [1.0] + [-1.0*S_ub[(s,t)]] ] ]
                c.linear_constraints.add(lin_expr=row, senses="L", rhs=[0.0])
                
                row = [ [ [y_minus[(s,t)]] + [yPrime[(s,t)]], [1.0] + [S_lb[(s,t)]] ] ]
                c.linear_constraints.add(lin_expr=row, senses="G", rhs=[S_lb[(s,t)]])

    #Add strengthened activation constraints
    for t in range(horizon):
        for relu in relus:
        
            inputs = []
            coefs = []
            for inp in inputNeurons[(relu)]:
                if inp in mappings:
                    if mappings[(inp)] in A:
                        if (A_lb[(mappings[(inp)],t)] >= 0.0 and weights[(inp,relu)] > 0.0) or (A_ub[(mappings[(inp)],t)] <= 0.0 and weights[(inp,relu)] < 0.0):
                            coefs.append(weights[(inp,relu)])
                            inputs.append(x[(mappings[(inp)],t)])
                        elif A_lb[(mappings[(inp)],t)] < 0.0 and A_ub[(mappings[(inp)],t)] > 0.0:
                            coefs.append(weights[(inp,relu)])
                            if weights[(inp,relu)] > 0.0:
                                inputs.append(x_plus[(mappings[(inp)],t)])
                            else:
                                inputs.append(x_minus[(mappings[(inp)],t)])

                    else:
                        if (S_lb[(mappings[(inp)],t)] >= 0.0 and weights[(inp,relu)] > 0.0) or (S_ub[(mappings[(inp)],t)] <= 0.0 and weights[(inp,relu)] < 0.0):
                            coefs.append(weights[(inp,relu)])
                            inputs.append(y[(mappings[(inp)],t)])
                        elif S_lb[(mappings[(inp)],t)] < 0.0 and S_ub[(mappings[(inp)],t)] > 0.0:
                            coefs.append(weights[(inp,relu)])
                            if weights[(inp,relu)] > 0.0:
                                inputs.append(y_plus[(mappings[(inp)],t)])
                            else:
                                inputs.append(y_minus[(mappings[(inp)],t)])
                else:
                    if weights[(inp,relu)] > 0.0:
                        coefs.append(weights[(inp,relu)])
                        inputs.append(z[(inp,t)])
                    
    
            row = [ [ inputs + [z[(relu,t)]] + [zPrime[(relu,t)]], coefs + [-1.0] + [bias[(relu)]] ] ]
            c.linear_constraints.add(lin_expr=row, senses="G", rhs=[0.0])

    return c

def encode_hd_milp_plan(domain, instance, horizon, sparsification, bound):
    
    bigM = 1000000.0
    
    inputNeurons, weights, bias, activationType = readDNN("./dnn/dnn_"+domain+"_"+instance+".txt")
    initial = readInitial("./translation/initial_"+domain+"_"+instance+".txt")
    goal = readGoal("./translation/goal_"+domain+"_"+instance+".txt")
    constraints = readConstraints("./translation/constraints_"+domain+"_"+instance+".txt")
    A, S, Aux, A_type, S_type, Aux_type = readVariables("./translation/pvariables_"+domain+"_"+instance+".txt")
    mappings = readMappings("./translation/mappings_"+domain+"_"+instance+".txt")
    
    relus = [relu for relu in inputNeurons.keys() if activationType[(relu)] == "relu"]
    outputs = [output for output in inputNeurons.keys() if activationType[(output)] == "linear" or activationType[(output)] == "step"]
    
    transitions = []
    if len(outputs) < len(S):
        transitions = readTransitions("./translation/transitions_"+domain+"_"+instance+".txt")
    
    if sparsification > 0.0:
        weights, bias = sparsifyDNN(sparsification, weights, bias, inputNeurons, mappings, relus, outputs)

    reward = readReward("./translation/reward_"+domain+"_"+instance+".txt")
    
    # CPLEX
    c = cplex.Cplex()

    # Set number of threads
    c.parameters.threads.set(1)

    # Initialize variables
    c, x, y, v, z, zPrime, vartypes, colnames = initialize_variables(c, A, S, Aux, relus, A_type, S_type, Aux_type, horizon)

    # Set global constraints
    c = encode_global_constraints(c, constraints, A, S, Aux, x, y, v, horizon)

    # Set initial state
    c = encode_initial_constraints(c, initial, y)

    # Set goal state
    c = encode_goal_constraints(c, goal, y, horizon)

    # Set node activations
    c = encode_activation_constraints(c, relus, bias, inputNeurons, mappings, weights, A, S, x, y, z, zPrime, bigM, horizon)

    # Predict the next state using DNNs
    c = encode_nextstate_constraints(c, outputs, bias, inputNeurons, mappings, weights, A, S, x, y, z, activationType, S_type, bigM, horizon)

    if bound == "True":
        # Set strengthened activation constraints
        c = encode_strengthened_activation_constraints(c, A, S, relus, bias, inputNeurons, mappings, weights, colnames, x, y, z, zPrime, horizon)

    if len(outputs) < len(S):
        # Set known transition function
        c = encode_known_transitions(c, transitions, A, S, Aux, x, y, v, horizon)

    # Reward function
    c = encode_reward(c, reward, colnames, A, S, Aux, x, y, v, horizon)

    # Set time limit
    #c.parameters.timelimit.set(3600.0)
    
    # Set optimality tolerance
    #c.parameters.mip.tolerances.mipgap.set(0.2)
    
    c.solve()

    #c.write("hd_milp_plan.lp")

    solution = c.solution
    
    print("")

    if solution.get_status() == solution.status.MIP_infeasible:
        print("No plans w.r.t. the given DNN exists.")
    elif solution.get_status() == solution.status.MIP_optimal:
        print("An optimal plan w.r.t. the given DNN is found:")
        
        solX = solution.get_values()
        #for s in S:
        #    print("%s at time %d by: %f " % (s,0,solX[y[(s,0)]]))
        for t in range(horizon):
            for a in A:
                print("%s at time %d by: %f " % (a,t,solX[x[(a,t)]]))
            #for s in S:
            #    print("%s at time %d by: %f " % (s,t+1,solX[y[(s,t+1)]]))
    elif solution.get_status() == solution.status.MIP_feasible or solution.get_status() == solution.status.MIP_abort_feasible or solution.get_status() == solution.status.MIP_time_limit_feasible or solution.get_status() == solution.status.optimal_tolerance:
        print("A plan w.r.t. the given DNN is found:")
        
        solX = solution.get_values()
        #for s in S:
        #    print("%s at time %d by: %f " % (s,0,solX[y[(s,0)]]))
        for t in range(horizon):
            for a in A:
                print("%s at time %d by: %f " % (a,t,solX[x[(a,t)]]))
            #for s in S:
            #    print("%s at time %d by: %f " % (s,t+1,solX[y[(s,t+1)]]))
    elif solution.get_status() == solution.status.MIP_abort_infeasible:
        print("Planning is interrupted by the user.")
    elif solution.get_status() == solution.status.MIP_time_limit_infeasible:
        print("Planning is terminated by the time limit without a plan.")
    else:
        print("Planning is interrupted. See the status message: %d" % solution.get_status())

    print("")

    return

def get_args():

    import sys
    argv = sys.argv
    
    myargs = {}

    for index, arg in enumerate(argv):
        if arg[0] == '-':
            myargs[arg] = argv[index+1]

    return myargs

if __name__ == '__main__':
    
    myargs = get_args()
    
    setDomain = False
    setInstance = False
    setHorizon = False
    setSparsification = False
    setBounds = False
    
    sparsification = "0.0"
    
    for arg in myargs:
        if arg == "-d":
            domain = myargs[(arg)]
            setDomain = True
        elif arg == "-i":
            instance = myargs[(arg)]
            setInstance = True
        elif arg == "-h":
            horizon = myargs[(arg)]
            setHorizon = True
        elif arg == "-s":
            sparsification = myargs[(arg)]
            setSparsification = True
        elif arg == "-b":
            bound = myargs[(arg)]
            setBounds = True

    if setDomain and setInstance and setHorizon and setBounds:
        encode_hd_milp_plan(domain, instance, int(horizon), float(sparsification), bound)
    elif not setDomain:
        print 'Domain is not provided.'
    elif not setInstance:
        print 'Instance is not provided.'
    elif not setHorizon:
        print 'Horizon is not provided.'
    else:
        print 'Bounding decision is not provided.'
