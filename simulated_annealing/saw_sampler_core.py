import networkit as nk
import numpy as np

def propose_path_mcmc(G,beta,start_node,end_node):
    """
    This function proposes a self avoiding path from start_node to end_node by sampling from an autoregressive process.

    The path is built starting from the start_node and adding nodes one by one until the end_node is reached.
    To satisfy the self avoiding constraint G changes as new nodes are added to the path: every time a new vertex u is added to the path, all the edges v->u for v a neighbour of u are removed. 
    When the function is finished executing, the graph G is returned to its original state.
    Together with the self avoiding path, also its probability under the proposal distribution is computed. 
    This probability is the product of the transition probabilities from one node to the next, and is used to compute the acceptance rate in the metropolis step.
    
    Args:
    G: networkit graph whose edges are weighted and directed. The cost of a path is the sum of the weights of the edges in the path.
    beta (float): inverse temperature parameter. WARNING: beta=0 is not allowed, however small values of beta are allowed.
    start_node (int): starting node of the path
    end_node (int): ending node of the path

    WARNING: the algorithm assumes that there is a path from start_node to end_node. If there is no path, the algorithm can produce unexpected results.

    Returns:
    path (list): list of nodes joined by edges in the graph G and sampled from the proposal Q. path is self avoiding: no node is visited more than once.
    path_log_prob (float): log probability of the path under the proposal distribution ( log Q(path) ).
    """
    BIG_DISTANCE=1e+200 #If the weighted distance from the start_node to the end_node is larger than this value, the algorithm will throw a warning, since it could mean that start_node and end_node are not connected in G.
    curr_node=start_node
    path=[start_node]
    removed_edges=[]
    #keeping track of the log probability, to be used for metropolis acceptance rate.
    path_log_prob=0
    #loop over path length
    while(curr_node!=end_node):
        #removing edges going into the last node in the path
        edges_to_remove=[]
        for v in G.iterInNeighbors(curr_node): #removing incoming edges
            edges_to_remove.append([v,curr_node, G.weight(v,curr_node)])
        for edge in edges_to_remove:
            G.removeEdge(edge[0],edge[1]) 
        removed_edges=removed_edges+edges_to_remove
        
        out_degree=G.degreeOut(curr_node)
        C_path_y=np.zeros(out_degree) #array containing the C coefficients for the current iteration.
        neigh_curr_node=np.zeros(out_degree, dtype=np.uintc) #neighbours of the current node
        i=0
        for y,w in G.iterNeighborsWeights(curr_node): #iterates over the out-neighbours of curr_node
            spsp = nk.distance.SPSP(G,[y]) #WARNING: if there is no path from y to e, the getDistance function will return 1.8e+308, which is virtually infinity, unless beta=0.
            # therefore if beta=0, the algorithm will not work.
            spsp.run()
            V_e_y=spsp.getDistance(y,end_node)
            C_path_y[i]=V_e_y+w
            neigh_curr_node[i]=y
            i=i+1
        
        #defining the transition probability
        min_C=np.min(C_path_y)
        if(min_C>BIG_DISTANCE):
            print("WARNING: the distance from node",curr_node,"to node",end_node,"is larger than",BIG_DISTANCE,". This could mean that there is no path from the start_node to the end_node.")

        C_path_y=C_path_y-min_C #regularizing
        P_trans=np.exp(-beta*C_path_y)   
        P_trans=P_trans/np.sum(P_trans) #normalize the probability
        idx_new_node=np.random.choice(a=len(P_trans),size=None, replace=True, p=P_trans)
        new_node=neigh_curr_node[idx_new_node]
        
        path.append(new_node)
        path_log_prob+=np.log(P_trans[idx_new_node])
        curr_node=new_node
        
    #restore removed edges so that G is not modified at the end. This avoids creating copies of G every time.
    for edge in removed_edges:
        G.addEdge(edge[0],edge[1])
        G.setWeight(edge[0],edge[1],edge[2])
            
    return path, path_log_prob


def compute_prop_path_log_prob(G, beta, path):
    """
    This function that computes the probability of a path under the proposal distribution Q.
    This function is used exclusively at initialization of the MCMC to compute the probability of the initial path under the proposal distribution. 
    The probability is needed for the first metropolis step. At subsequent time steps the proposal probability is computed directly in 'propose_path_mcmc'.
    
    Args:
    G: networkit graph whose edges are weighted and directed. The cost of a path is the sum of the weights of the edges in the path.
    beta (float): inverse temperature parameter. WARNING: beta=0 is not allowed, however small values of beta are allowed.
    path (list): list of nodes joined by edges in the graph G.

    WARNING: the algorithm will not perform checks on the validity of the path, so it is assumed that the path is valid

    Returns:
    path_log_prob (float): log probability of the path under the proposal distribution ( log Q(path) ).
    """
    end_node=path[-1]
    removed_edges=[]
    #keeping track of the log probability, to be used for metropolis acceptance rate.
    path_log_prob=0
    #loop over path length
    for k in range(len(path)-1):
        curr_node=path[k]
        #removing edges going into the last node in the path
        edges_to_remove=[]
        for v in G.iterInNeighbors(curr_node): #removing incoming edges
            edges_to_remove.append([v,curr_node, G.weight(v,curr_node)])
        for edge in edges_to_remove:
            G.removeEdge(edge[0],edge[1]) 
        removed_edges=removed_edges+edges_to_remove
        
        out_degree=G.degreeOut(curr_node)
        C_path_y=np.zeros(out_degree) #array containing the C coefficients for the current iteration.
        neigh_curr_node=np.zeros(out_degree, dtype=np.uintc) #neighbours of the current node
        i=0
        for y,w in G.iterNeighborsWeights(curr_node): #iterates over the out-neighbours of curr_node
            spsp = nk.distance.SPSP(G,[y]) #WARNING: if there is no path from y to e, the getDistance function will return 1.8e+308, which is virtually infinity, unless beta=0.
            # therefore if beta=0, the algorithm will not work.
            spsp.run()
            V_e_y=spsp.getDistance(y,end_node) 
            C_path_y[i]=V_e_y+w
            neigh_curr_node[i]=y
            i=i+1
        
        #defining the transition probability
        C_path_y=C_path_y-np.min(C_path_y) #regularizing
        log_P_trans=-beta*C_path_y
        #P_trans=P_trans/np.sum(P_trans) #normalize the probability
        new_node=path[k+1]
        idx_new_node=np.where(neigh_curr_node==new_node)[0]
        path_log_prob+=log_P_trans[idx_new_node]-np.log(np.sum(np.exp(log_P_trans)))
        
    #restore removed edges so that G is not modified at the end. This avoids creating copies of G every time.
    for edge in removed_edges:
        G.addEdge(edge[0],edge[1])
        G.setWeight(edge[0],edge[1],edge[2])
    return path_log_prob

        
def mcmcm_saw(beta, G, start_node, end_node, tmax, init_path=None):
    """
    Function that implements a whole MCMC Metropolis simulation.
    
    The simulation is used to sample several self-avoiding paths from the distribution P, of paths going from start_node to end_node, in the graph G.

    Args:
    beta (float): inverse temperature parameter. WARNING: beta=0 is not allowed, however small values of beta are allowed.
    G: networkit graph whose edges are weighted and directed. The cost of a path is the sum of the weights of the edges in the path.
    start_node (int): starting node of the path
    end_node (int): ending node of the path
    tmax (int): length of the simulation
    init_path (list): initial path to start the simulation. If None, the algorithm will start from the shortest path from start_node to end_node.

    Returns:
    paths (list): list of paths sampled during the simulation. Each path is a list of nodes joined by edges in the graph G. The length of the list is tmax+1, since the initial path is also included.
    num_accepted (int): number of accepted proposals during the simulation. The acceptance rate is num_accepted/tmax.
    """
    if(init_path == None):
        dijkstra = nk.distance.Dijkstra(G, source=start_node)
        dijkstra.run()
        path = dijkstra.getPath(end_node)
    else:
        path=init_path.copy()
    path_log_prob=compute_prop_path_log_prob(G, beta, path)
    paths=[path.copy()]
    path_energy=sum([G.weight(path[k],path[k+1]) for k in range(len(path)-1)])
    num_accepted=0
    log_random_numbers=np.log(1-np.random.uniform(low=0.0, high=1.0, size=tmax))#precompute the randomness.
    for t in range(tmax):
        prop_path, prop_path_log_prob=propose_path_mcmc(G, beta, start_node, end_node) #the function propose_path_mcmc does not alter G
        prop_path_energy=sum([G.weight(prop_path[k],prop_path[k+1]) for k in range(len(prop_path)-1)]) #compute the energy of the proposed path
        log_p_acc=beta*(path_energy-prop_path_energy)+(path_log_prob-prop_path_log_prob) #compute the log of the acceptance probability
        if(log_random_numbers[t]<log_p_acc): #accept the proposal
            num_accepted+=1
            path=prop_path.copy() #update the path
            path_energy=prop_path_energy
            path_log_prob=prop_path_log_prob
        paths.append(path.copy())
        
    return paths, num_accepted/tmax