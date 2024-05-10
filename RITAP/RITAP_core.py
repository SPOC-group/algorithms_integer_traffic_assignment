import numpy as np
import copy
import networkit as nk
import time
import networkx as nx

def copy_graph(G, reset_weights=False, weight_value=0):
    """
    Returns a copy of a networkit graph G
    
    Args:
    G: networkit graph
    reset_weights: bool, if True, the weights of the graph are set to 'weight_value'
    weight_value: float, the value to which the weights are set if reset_weights==True

    Returns:
    G_copy: a copy of G
    """
    N = G.numberOfNodes()
    G_copy = nk.graph.Graph(
        n=G.numberOfNodes(),
        weighted=G.isWeighted(),
        directed=G.isDirected(),
        edgesIndexed=False,
    )
    for u, v in G.iterEdges():
        G_copy.addEdge(u, v)
        if G.isWeighted():
            if reset_weights:
                G_copy.setWeight(u, v, weight_value)
            else:
                G_copy.setWeight(u, v, G.weight(u, v))
    return G_copy


def starts_ends_to_origin_bush(start_nodes, end_nodes):
    """
    This function converts the two lists start_nodes and end_nodes into a different data structure that collects all the paths with a commom start node.

    A bush is a set of start-end pairs with a common start node. The data structure to represent many bushes is the following
    {s1 : {e1_s1 : w_s1e1, e2_s1 : w_s1e2,...}, s2 : {e1_s2 : w_s2e1, e2_s2 : w_s2e2,...}, ...}
    s are the start nodes
    e_s are the various end nodes corresponding to a start node s
    w_se is the number of paths going from s to e.
    The advantage of this data structure is the following: dijkstra computes the shortest paths from one start node to all possible end nodes. Therefore grouping the paths by start node we reduce the number fo calls to dijkstra.
    The ordering of the paths is lost in this new representation.

    Args:
    start_nodes: list of integers
    end_nodes: list of integers

    Returns:
    bushes: dictionary of dictionaries of integers
    """
    bushes = {}

    for start, end in zip(start_nodes, end_nodes):
        if start in bushes:
            if end in bushes[start]:
                bushes[start][end] = bushes[start][end] + 1
            else:
                bushes[start][end] = 1
        else:
            bushes[start] = {end: 1}

    return bushes



def FW_traffic_assignment_single_commodity(bushes, G, lr, tmax, phi, deriv_phi, rtol=1e-7, rel_opt_gap_thr=0):
    """
    Uses the Frank-Wolfe algorithm to compute the continuous traffic equilibrium that minimizes H(I)=sum_e phi(I_e), where I_e is the total traffic on edge e.
    
    Normally the FW algorithm does not keep track of the paths through which the flow is routed, instead it only keeps track of the total traffic on each edge.
    This implementation however keeps track of the paths as well. This is needed in order to project the TAP paths onto ITAP ones.
    This implementation is for undirected graphs only. Also the nonlinearity phi is assumed to be the same on every edge of the graph. The graph is assumed to be connected.
    
    Args:
    bushes: dictionary of dictionaries of integers. This data structure is the output of the function starts_ends_to_origin_bush. It is a dictionary of dictionaries of integers. The outer dictionary has as keys the start nodes, while the inner dictionaries have as keys the end nodes. The values of the inner dictionaries are the number of paths going from the start node to the end node.
    G: weighted networkit undirected graph. G's weights will be modified by the algorithm
    lr (list): learning rate of the Frank-Wolfe algorithm. To have convergence, lr[t] shoudl go to zero when t becomes large. It is expected that len(lr)=tmax
    tmax (int): maximum number of Frank-Wolfe iterations
    phi (function): nonlinearity in the Hamiltonian
    phi_deriv (function): derivative of phi
    rtol (float): relative tolerance before declaring convergence. if |H(t)-H(t-1)|/H(t-1)]<rtol*lr[t], then the algorithm halts. H(t) here is the value of the energy after t iterations.
    rel_opt_gap_thr: alternative halting condition based on the otimality gap. When |optimality_gaps[t]|/H(t)<rel_opt_gap_thr, the algorithm halts. optimality_gaps[t] in turn is an upper bound to H[t]-H_*, where H_* is the energy of the optimal solution. WARNING:This is only valid when phi is convex. If phi is not convex this quantity is meaningless, and one should set rel_opt_gap_thr to zero.

    Returns:
    G_traf: a networkit weighted undirected graph. The weight on each edge e is equal to the total flow I_e on that edge
    G: a networkit weighted undirected graph. The weight on each edge e is equal to phi'(I_e), where I_e is the total traffic on edge e and phi' is the derivative of phi.
    dict_paths: this dictionary contains each commodity 's flow on the graph. The keys of the dictionary are tuples of the form (start,end), (edge[0],edge[1]). The first pair of keys indexes the top level dictonary, while the second pair of keys indexes the second level dictionary. dict_paths[start,end][edge[0],edge[1]] is the flow of the commodity that goes from start to end and traverses the edge (edge[0],edge[1]).
    energies (list): the list of values of H encountered at every time step of the algorithm.
    optimality_gaps (list): if phi is convex, then energies[t]-optimality_gaps[t] is a lower bound to the energy of the optimal solution. THIS IS EXCLUSIVELY VALID WHEN PHI IS CONVEX, otherwise this quantity is meaningless.
    conv_flag (bool): if True the algorithm has halted because it satisfied one of the convergence conditions, if False it halted when it finished tmax iterations
    t (integer): the iteration number at which the algorithm converged
    """

    energies = []
    optimality_gaps = []

    # reset weights in G
    for u, v in G.iterEdges():
        G.setWeight(u, v, 1)

    dict_paths = {} # this nested dictionary contains the flows for every start-end pair. If there are multiple paths with the same start-end pair, their combined flow is stored in the same entry of the dictionary.
    # keys of this dictionaries are tuples with this structure: (start,end), (edge[0],edge[1]). The first pair of keys indexes the top level dictonary, while the second pair of keys indexes the second level dictionary.
    # start is the starting node , end is the final node, edge[0], edge[1] are the two nodes joined by the edge.
    G_traf = copy_graph(G, reset_weights=True, weight_value=0)
    for start in bushes:
        dijkstra = nk.distance.Dijkstra(G, source=start)
        dijkstra.run()
        for end in bushes[start]:
            dict_paths[start, end] = {}
            path = dijkstra.getPath(end)
            for k in range(len(path) - 1):
                dict_paths[start, end][path[k], path[k + 1]] = bushes[start][end]
                G_traf.setWeight(path[k],path[k + 1],G_traf.weight(path[k], path[k + 1]) + bushes[start][end])

    # auxiliary graph where the all-or-nothing (aon) flows will be stored. These are called all-or-nothing because all the flow between a start and an end node is routed through a single path.
    G_traf_aon = copy_graph(G, reset_weights=True, weight_value=0)
    conv_flag = False
    for t in range(tmax):
        # measure_energy
        total_energy = sum([phi(w) for _, _, w in G_traf.iterEdgesWeights()])
        energies.append(total_energy)
        # convergence condition
        if(t>1):
            if (abs(energies[t] - energies[t - 1]) / energies[t - 1] < rtol * lr[t] or abs(opt_gap) / energies[t - 1] < rel_opt_gap_thr):
                conv_flag = True
                optimality_gaps.append(opt_gap)
                break

        # compute the gradient
        for u, v in G.iterEdges():
            G.setWeight(u, v, deriv_phi(G_traf.weight(u, v)))
            G_traf_aon.setWeight(u, v, 0)

        for key in dict_paths: 
            for key2 in dict_paths[key]:
                dict_paths[key][key2] = dict_paths[key][key2] * (1 - lr[t])

        # compute all-or-nothing solution
        # use dijkstra to minimize
        for start in bushes:
            dijkstra = nk.distance.Dijkstra(G, source=start)
            dijkstra.run()
            for end in bushes[start]:
                path = dijkstra.getPath(end)
                for k in range(len(path) - 1):
                    G_traf_aon.setWeight(path[k],path[k + 1],G_traf_aon.weight(path[k], path[k + 1]) + bushes[start][end])
                    if (path[k], path[k + 1]) in dict_paths[start, end]:
                        dict_paths[start, end][path[k], path[k + 1]] = (dict_paths[start, end][path[k], path[k + 1]]+ lr[t] * bushes[start][end])
                    else:
                        dict_paths[start, end][path[k], path[k + 1]] = (lr[t] * bushes[start][end])

        # making a step towards G_traf_aon
        opt_gap = 0
        for u, v in G_traf.iterEdges():
            opt_gap -= G.weight(u, v) * (G_traf_aon.weight(u, v) - G_traf.weight(u, v))
            G_traf.setWeight(u,v,(1 - lr[t]) * G_traf.weight(u, v) + lr[t] * G_traf_aon.weight(u, v))

        optimality_gaps.append(opt_gap)

    return G_traf, G, dict_paths, energies, optimality_gaps, conv_flag, t


def single_flow_to_paths(G_traf, start, end, flow_eps=0, maxiter=None):
    """
    Given a single commodity flow, between a source node 'start' and an end node 'end', this algorithm finds a path decomposition of this flow.
    In other words, given the distriution of the flow on the edges of the graph, the algorithm finds the paths that carry the flow. The path decomposition is not unique.
    The algorithm iteratively finds the path carrying the most flow, and then subtracts the flow carried by that path from the total flow. The algorithm stops when the flow on the edges is smaller than 'flow_eps' or when the maximum number of iterations is reached.
    
    Args:
    G_traf: weighted networkit graph containing the flow (of the single commodity, i.e., single start-end pair )on each edge of the graph
    start (int): start node of the flow (source node)
    end(int): end node of the flow (sink node)
    flow_eps: (float) flows smaller than flow_eps will be set to zero.
    maxiter: (int) maximum number of iterations default is None, corresponding to infinite limit.

    Returns:
    path_list: a list in which every element is a path. A path is a list containing the sequence of nodes traversed by the path, starting from the start node and finishing with the end node.
    path_traf_list: for every path this list specifies the traffic carried by that path. path_list[i] brings flow path_traf_list[i].
    """

    BIGNUM = 1e100
    incr = 1
    if maxiter is None:
        maxiter = 1
        incr = 0

    a = 0.001  # the smaller a, the more likely it is that one finds paths carrying more traffic (as opposed to shorter paths) first (when runninng the loop)
    G_aux = copy_graph(
        G_traf, reset_weights=True, weight_value=BIGNUM
    )  # edges with a lot of flow in G_traf will have a small cost in G_aux. We use dijkstra on G_aux to select the paths. This way we will first select the paths carrying the most flow.
    for u, v, w in G_traf.iterEdgesWeights():
        if w > flow_eps:
            G_aux.setWeight(u, v, 1 / (w + a))
    if start == end:
        print("ERROR start node should be different from end node")
        return 0

    path_list = []
    path_traf_list = []
    iter_count = 0
    while iter_count < maxiter:
        dijkstra = nk.distance.Dijkstra(G_aux, source=start)
        dijkstra.run()
        path = dijkstra.getPath(end)
        path_traf = min([G_traf.weight(path[k], path[k + 1]) for k in range(len(path) - 1)])
        if dijkstra.distance(end) > 0.5 * BIGNUM or path_traf < flow_eps:
            break

        for k in range(len(path) - 1):
            G_traf.setWeight(path[k], path[k + 1], G_traf.weight(path[k], path[k + 1]) - path_traf)  # subtracting the contribution of the path from the total flow
            if G_traf.weight(path[k], path[k + 1]) <= flow_eps:
                G_aux.setWeight(path[k], path[k + 1], BIGNUM)
                G_traf.setWeight(path[k], path[k + 1], 0)

            else:
                G_aux.setWeight(path[k], path[k + 1], 1 / (a + G_traf.weight(path[k], path[k + 1])))

        path_list.append(path.copy())
        path_traf_list.append(path_traf)
        iter_count += incr

    path_traf_list, path_list = (list(t) for t in zip(*sorted(zip(path_traf_list, path_list), reverse=True)))
    return path_list, path_traf_list


def flows_to_paths(G_traf, dict_paths, bushes, flow_eps=1e-12):
    """
    Converts the flow on the edges of the graph into paths. The paths are the paths that carry the flow. The path decomposition is not unique.
    Internally this functino calls the function single_flow_to_paths for every start-end pair.

    Args:
    G_traf: networkit graph whose edges are weighted with I_e, the total flow through the edge.
    dict_paths: dictionary containing the traffic on each edge for every commodity. dict_paths[start,end]={e1: f1,e2:f2,...} where start,end are the starting and ending point of the path. e1=(e10,e11) is an edge is the graph, f1,f2,... are the flows on the respective edges.
    bushes: bushes representation of the demand matrix (see starts_ends_to_origin_bush)
    flow_eps: flows smaller that flow_eps are set to zero

    Returns:
    paths_dict: paths_dict[start,end]=[[p1,p2,...],[t1,t2,..]], where paths p1,p2,... are the paths between start and end that carry the traffic. t1,t2,... are the traffic on the respective paths. each of p1,p2,... is a list contai ing the nodes traversed by the path.
    """
    paths_dict = {}
    for start in bushes:
        for end in bushes[start]:
            G_traf_se = copy_graph(G_traf, reset_weights=True, weight_value=0)  # graph storing the flow for all the paths with common start point and end point
            for u, v in dict_paths[start, end]:
                if (dict_paths[start, end][u, v] > flow_eps):  # setting the flow to zero if it's too small
                    G_traf_se.setWeight(u, v, dict_paths[start, end][u, v])
            paths_se, traffic_paths_se = single_flow_to_paths(G_traf_se, start, end, flow_eps=flow_eps)
            paths_dict[start, end] = [copy.deepcopy(paths_se), traffic_paths_se.copy()]

    return paths_dict


def continuous_to_max_traf_integer_paths(paths_dict, bushes):
    """
    Computes the integer paths starting from the paths carrying a noninteger traffic. 
    
    The algorithm first assigns the integer part of the traffic to the paths. Then it assigns the remaining flow to paths in order of decreasing fractional flow.
    The algorithm is conceived to handle correctly the case when there are multiple paths with the same start and end nodes.

    Args:
    paths_dict: nested dictionary (usually outputted by flows_to_paths). Its structure is paths_dict[start,end]=[[p1,p2,...],[t1,t2,..]], where paths p1,p2,... are the paths between start and end that carry the traffic. t1,t2,... are the traffic on the respective paths. each of p1,p2,... is a list contai ing the nodes traversed by the path.
    bushes: bushes representation of the demand matrix

    Returns:
    a bush (nested dictionary whose keys are the start and end node) integer_paths_bush, such that integer_paths_bush[start][end]=[[pi_1,...,pi_R],[tINT_1,...,tINT_R], [t_1,...,t_R]] with pi_i paths from start to end carrying an integer amount of flow tINT_i and originally carrying an amount of flow t_i in the continuous solution
    """
    integer_paths_dict ={} #bush (nested dictionary) containing the paths and flow for each start-end pair
    for start,end in paths_dict:
        if integer_paths_dict.get(start) is None:
            integer_paths_dict[start]={}
        path_list=copy.deepcopy(paths_dict[start, end][0])
        path_traf_list=copy.copy(paths_dict[start,end][1])
        integer_paths_dict[start][end]=[[],[],[]]
        allocated_flow=0
        for i,traf in enumerate(path_traf_list):
            if(traf>=1): #adding the paths with more than one unit of flow
                integer_paths_dict[start][end][0].append(path_list[i])
                integer_paths_dict[start][end][1].append(int(traf))
                integer_paths_dict[start][end][2].append(int(traf))
                allocated_flow+=int(traf)
                path_traf_list[i]=path_traf_list[i]%1 #keeping the fractional part

        if(allocated_flow<bushes[start][end]): #now adding the fractional parts if necessary
            path_traf_list, path_list = (list(t) for t in zip(*sorted(zip(path_traf_list, path_list), reverse=True))) # sorting paths in order of decreasing fractional traffic
            for k in range(bushes[start][end]-allocated_flow):
                if(path_list[k] in integer_paths_dict[start][end][0]):#if the path is already present in the list we increment its 
                    idx=integer_paths_dict[start][end][0].index(path_list[k])
                    integer_paths_dict[start][end][1][idx]+=1
                    integer_paths_dict[start][end][2][idx]+=path_traf_list[k]

                else:# if the path is not present we append it to the list of paths
                    integer_paths_dict[start][end][0].append(path_list[k])
                    integer_paths_dict[start][end][1].append(1)
                    integer_paths_dict[start][end][2].append(path_traf_list[k])

    return integer_paths_dict

def FW_traffic_assignment_single_commodity_directed(bushes, G, lr, tmax, phi, deriv_phi, rtol=1e-7, rel_opt_gap_thr=0):
    """
    Uses the Frank-Wolfe algorithm to compute the continuous traffic equilibrium that minimizes H(I)=sum_e phi(I_e), where I_e is the total traffic on edge e.

    
    Normally the FW algorithm does not keep track of the paths through which the flow is routed, instead it only keeps track of the total traffic on each edge.
    This implementation however keeps track of the paths as well. This is needed in order to project the TAP paths onto ITAP ones.
    This implementation is for directed graphs only. Also the nonlinearity phi is assumed to be the same on every edge of the graph. The graph is assumed to be connected.
    
    Args:
    bushes: dictionary of dictionaries of integers. This data structure is the output of the function starts_ends_to_origin_bush. It is a dictionary of dictionaries of integers. The outer dictionary has as keys the start nodes, while the inner dictionaries have as keys the end nodes. The values of the inner dictionaries are the number of paths going from the start node to the end node.
    G: weighted networkit directed graph. G's weights will be modified by the algorithm
    lr (list): learning rate of the Frank-Wolfe algorithm. To have convergence, lr[t] shoudl go to zero when t becomes large. It is expected that len(lr)=tmax
    tmax (int): maximum number of Frank-Wolfe iterations
    phi (function): nonlinearity in the Hamiltonian
    phi_deriv (function): derivative of phi
    rtol (float): relative tolerance before declaring convergence. if |H(t)-H(t-1)|/H(t-1)]<rtol*lr[t], then the algorithm halts. H(t) here is the value of the energy after t iterations.
    rel_opt_gap_thr: alternative halting condition based on the otimality gap. When |optimality_gaps[t]|/H(t)<rel_opt_gap_thr, the algorithm halts. optimality_gaps[t] in turn is an upper bound to H[t]-H_*, where H_* is the energy of the optimal solution. WARNING:This is only valid when phi is convex. If phi is not convex this quantity is meaningless, and one should set rel_opt_gap_thr to zero.

    Returns:
    G_traf: a networkit weighted undirected graph. The weight on each edge e is equal to the total flow I_e on that edge
    G: a networkit weighted undirected graph. The weight on each edge e is equal to phi'(I_e), where I_e is the total traffic on edge e and phi' is the derivative of phi.
    dict_paths: this dictionary contains each commodity 's flow on the graph. The keys of the dictionary are tuples of the form (start,end), (edge[0],edge[1]). The first pair of keys indexes the top level dictonary, while the second pair of keys indexes the second level dictionary. dict_paths[start,end][edge[0],edge[1]] is the flow of the commodity that goes from start to end and traverses the edge (edge[0],edge[1]).
    energies (list): the list of values of H encountered at every time step of the algorithm.
    optimality_gaps (list): if phi is convex, then energies[t]-optimality_gaps[t] is a lower bound to the energy of the optimal solution. THIS IS EXCLUSIVELY VALID WHEN PHI IS CONVEX, otherwise this quantity is meaningless.
    conv_flag (bool): if True the algorithm has halted because it satisfied one of the convergence conditions, if False it halted when it finished tmax iterations
    t (integer): the iteration number at which the algorithm converged
    """

    # convert start_nodes, end_nodes to bush data structure
    #N = G.numberOfNodes()
    #M = len(start_nodes)
    energies = []
    optimality_gaps = []

    # reset weights in G
    for u, v in G.iterEdges():
        G.setWeight(u, v, 1)

    dict_paths = {} # this nested dictionary contains the flows for every start-end pair. If there are multiple paths with the same start-end pair, their combined flow is stored in the same entry of the dictionary.
    # keys of this dictionaries are tuples with this structure: (start,end), (edge[0],edge[1]). The first pair of keys indexes the top level dictonary, while the second pair of keys indexes the second level dictionary.
    # start is the starting node , end is the final node, edge[0], edge[1] are the two nodes joined by the edge.
    G_traf = copy_graph(G, reset_weights=True, weight_value=0)
    for start in bushes:
        dijkstra = nk.distance.Dijkstra(G, source=start)
        dijkstra.run()
        for end in bushes[start]:
            dict_paths[start, end] = {}
            path = dijkstra.getPath(end)
            for k in range(len(path) - 1):
                dict_paths[start, end][path[k], path[k + 1]] = bushes[start][end]
                G_traf.setWeight(path[k],path[k + 1],G_traf.weight(path[k], path[k + 1]) + bushes[start][end])

    # auxiliary graph where the all-or-nothing (aon) flows will be stored. These are called all-or-nothing because all the flow between a start and an end node is routed through a single path.
    G_traf_aon = copy_graph(G, reset_weights=True, weight_value=0)
    conv_flag = False
    for t in range(tmax):
        # measure_energy
        total_energy = sum([phi(w) for _, _, w in G_traf.iterEdgesWeights()])
        energies.append(total_energy)
        # convergence condition
        if(t>1):
            if (abs(energies[t] - energies[t - 1]) / energies[t - 1] < rtol * lr[t] or abs(opt_gap) / energies[t - 1] < rel_opt_gap_thr):
                conv_flag = True
                optimality_gaps.append(opt_gap)
                break

        # compute the gradient
        for u, v in G.iterEdges():
            G.setWeight(u, v, deriv_phi(G_traf.weight(u, v)))
            G_traf_aon.setWeight(u, v, 0)

        for key in dict_paths: 
            for key2 in dict_paths[key]:
                dict_paths[key][key2] = dict_paths[key][key2] * (1 - lr[t])

        # compute all-or-nothing solution
        # use dijkstra to minimize
        for start in bushes:
            dijkstra = nk.distance.Dijkstra(G, source=start)
            dijkstra.run()
            for end in bushes[start]:
                path = dijkstra.getPath(end)
                for k in range(len(path) - 1):
                    G_traf_aon.setWeight(path[k],path[k + 1],G_traf_aon.weight(path[k], path[k + 1]) + bushes[start][end])
                    if (path[k], path[k + 1]) in dict_paths[start, end]:
                        dict_paths[start, end][path[k], path[k + 1]] = (dict_paths[start, end][path[k], path[k + 1]]+ lr[t] * bushes[start][end])
                    else:
                        dict_paths[start, end][path[k], path[k + 1]] = (lr[t] * bushes[start][end])

        # making a step towards G_traf_aon
        opt_gap = 0
        for u, v in G_traf.iterEdges():
            opt_gap -= G.weight(u, v) * (G_traf_aon.weight(u, v) - G_traf.weight(u, v))
            G_traf.setWeight(u,v,(1 - lr[t]) * G_traf.weight(u, v) + lr[t] * G_traf_aon.weight(u, v))

        optimality_gaps.append(opt_gap)

    return G_traf, G, dict_paths, energies, optimality_gaps, conv_flag, t


def FW_traffic_assignment_single_commodity_param_phi(bushes, G, lr, tmax, phi, deriv_phi, param_phi_dict, rtol=1e-7, rel_opt_gap_thr=0,first_thru_node=0):
    """
    Uses the Frank-Wolfe algorithm to compute the continuous traffic equilibrium that minimizes H(I)=sum_e phi(I_e), where I_e is the total traffic on edge e.

    
    Normally the FW algorithm does not keep track of the paths through which the flow is routed, instead it only keeps track of the total traffic on each edge.
    This implementation however keeps track of the paths as well. This is needed in order to project the TAP paths onto ITAP ones.
    This implementation is for directed graphs only. Also, differently from other implementations, the nonlinearity phi can depend on the edge in the graph.
    Phi depends on the edge through a number of parameters. The parameters are stored in the dictionary param_phi_dict. The keys of the dictionary are the edges of the graph, while the values are lists containing the parameters of the nonlinearity function on that edge.
    
    The graph is assumed to be connected.
    
    Args:
    bushes: dictionary of dictionaries of integers. This data structure is the output of the function starts_ends_to_origin_bush. It is a dictionary of dictionaries of integers. The outer dictionary has as keys the start nodes, while the inner dictionaries have as keys the end nodes. The values of the inner dictionaries are the number of paths going from the start node to the end node.
    G: weighted networkit directed graph. G's weights will be modified by the algorithm
    lr (list): learning rate of the Frank-Wolfe algorithm. To have convergence, lr[t] shoudl go to zero when t becomes large. It is expected that len(lr)=tmax
    tmax (int): maximum number of Frank-Wolfe iterations
    phi (function): nonlinearity in the Hamiltonian. phi takes as input an integer and a list of parameters. The integer is the flow on the edge, while the list of parameters is the list of parameters of the nonlinearity function on that edge.
    phi_deriv (function): derivative of phi. deriv_phi takes as input an integer and a list of parameters. The integer is the flow on the edge, while the list of parameters is the list of parameters of the nonlinearity function on that edge.
    param_BPR_dict: dictionary having edges as keys and lists as values. Every list contains the parameters of the BPR function on that edge
    rtol (float): relative tolerance before declaring convergence. if |H(t)-H(t-1)|/H(t-1)]<rtol*lr[t], then the algorithm halts. H(t) here is the value of the energy after t iterations.
    rel_opt_gap_thr: alternative halting condition based on the otimality gap. When |optimality_gaps[t]|/H(t)<rel_opt_gap_thr, the algorithm halts. optimality_gaps[t] in turn is an upper bound to H[t]-H_*, where H_* is the energy of the optimal solution. WARNING:This is only valid when phi is convex. If phi is not convex this quantity is meaningless, and one should set rel_opt_gap_thr to zero.

    Returns:
    G_traf: a networkit weighted undirected graph. The weight on each edge e is equal to the total flow I_e on that edge
    G: a networkit weighted undirected graph. The weight on each edge e is equal to phi'(I_e), where I_e is the total traffic on edge e and phi' is the derivative of phi.
    dict_paths: this dictionary contains each commodity 's flow on the graph. The keys of the dictionary are tuples of the form (start,end), (edge[0],edge[1]). The first pair of keys indexes the top level dictonary, while the second pair of keys indexes the second level dictionary. dict_paths[start,end][edge[0],edge[1]] is the flow of the commodity that goes from start to end and traverses the edge (edge[0],edge[1]).
    energies (list): the list of values of H encountered at every time step of the algorithm.
    optimality_gaps (list): if phi is convex, then energies[t]-optimality_gaps[t] is a lower bound to the energy of the optimal solution. THIS IS EXCLUSIVELY VALID WHEN PHI IS CONVEX, otherwise this quantity is meaningless.
    conv_flag (bool): if True the algorithm has halted because it satisfied one of the convergence conditions, if False it halted when it finished tmax iterations
    t (integer): the iteration number at which the algorithm converged
    """

    energies = []
    optimality_gaps = []

    # reset weights in G to free flow travel times
    for u, v in G.iterEdges():
        G.setWeight(u, v,  param_phi_dict[u,v][1])
    start_time=time.time()

    dict_paths = {} # this nested dictionary contains the flows for every start-end pair. If there are multiple paths with the same start-end pair, their combined flow is stored in the same entry of the dictionary.
    # keys of this dictionaries are tuples with this structure: (start,end), (edge[0],edge[1]). The first pair of keys indexes the top level dictonary, while the second pair of keys indexes the second level dictionary.
    # start is the starting node , end is the final node, edge[0], edge[1] are the two nodes joined by the edge.
    G_traf = copy_graph(G, reset_weights=True, weight_value=0)

    for start in bushes:
        dijkstra = nk.distance.Dijkstra(G, source=start)
        dijkstra.run()
        for end in bushes[start]:
            dict_paths[start, end] = {}
            path = dijkstra.getPath(end)
            for k in range(len(path) - 1):
                dict_paths[start, end][path[k], path[k + 1]] = bushes[start][end]
                G_traf.setWeight(path[k],path[k + 1],G_traf.weight(path[k], path[k + 1]) + bushes[start][end])

    # auxiliary graph where the all-or-nothing (aon) flows will be stored. These are called all-or-nothing because all the flow between a start and an end node is routed through a single path.
    G_traf_aon = copy_graph(G, reset_weights=True, weight_value=0)
    conv_flag = False
    for t in range(tmax):
        # measure_energy
        #if(t%20==0):
        #    print(t, time.time()-start_time)
        total_energy = sum([phi(w,param_phi_dict[e1,e2]) for e1, e2, w in G_traf.iterEdgesWeights()])
        energies.append(total_energy)
        # convergence condition
        if t > 1 and (abs(energies[t] - energies[t - 1]) / energies[t - 1] < rtol * lr[t] or abs(opt_gap) / energies[t - 1] < rel_opt_gap_thr):
            conv_flag = True
            optimality_gaps.append(opt_gap)
            break

        # compute the gradient
        for u, v in G.iterEdges():
            G.setWeight(u, v, deriv_phi(G_traf.weight(u, v),param_phi_dict[u,v]))
            G_traf_aon.setWeight(u, v, 0)

        for key in dict_paths: 
            for key2 in dict_paths[key]:
                dict_paths[key][key2] = dict_paths[key][key2] * (1 - lr[t])

        # compute all-or-nothing solution
        # use dijkstra to minimize
        for start in bushes:
            dijkstra = nk.distance.Dijkstra(G, source=start)
            dijkstra.run()
            for end in bushes[start]:
                path = dijkstra.getPath(end)
                for k in range(len(path) - 1):
                    G_traf_aon.setWeight(path[k],path[k + 1],G_traf_aon.weight(path[k], path[k + 1]) + bushes[start][end])
                    if (path[k], path[k + 1]) in dict_paths[start, end]:
                        dict_paths[start, end][path[k], path[k + 1]] = (dict_paths[start, end][path[k], path[k + 1]]+ lr[t] * bushes[start][end])
                    else:
                        dict_paths[start, end][path[k], path[k + 1]] = (lr[t] * bushes[start][end])

        # making a step towards G_traf_aon
        opt_gap = 0
        for u, v in G_traf.iterEdges():
            opt_gap -= G.weight(u, v) * (G_traf_aon.weight(u, v) - G_traf.weight(u, v))
            G_traf.setWeight(u,v,(1 - lr[t]) * G_traf.weight(u, v) + lr[t] * G_traf_aon.weight(u, v))

        optimality_gaps.append(opt_gap)

    return G_traf, G, dict_paths, energies, optimality_gaps, conv_flag, t