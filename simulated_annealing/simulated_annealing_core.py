import numpy as np
import networkit as nk
import saw_sampler_core as saw


def annealed_optimizer(G,start_nodes,end_nodes,beta_schedule,phi,mcmc_steps=10,mcmc_seed=None,quiet=True, conv_iters=5):
    """
    Simulated annealing based optimizer for the multiple path problem. 

    Args:
    G: networkit.graph (directed or undirected)
    start_nodes: list of integers
    end_nodes: list of integers
    beta_schedule: list of floats representing the annealing schedule
    phi: function, nonlinearity in the Hamiltonian
    mcmc_steps: int, number of MCMC steps to perform for each path. This is the number fo Metropolis steps to be performed every time a path is sampled
    mcmc_seed: int, seed for the random number generator
    quiet: bool, if True, the function does not print anything
    conv_iters: int, number of iterations without energy change to consider the algorithm converged
    
    Returns:
    paths: list of lists of integers, the optimal paths found by the algorithm
    flag_conv: bool, True if the algorithm converged before reaching the end of the annealing schedule, False otherwise


    The initial paths are computed using Dijkstra's algorithm, in other words the paths are initialized to be shortest paths with respect to the topological distance on G.
    In each iteration, an mcmc (which draws 'mcmc_steps' samples for each path) is performed for every path: this is the way paths are updated.
    If the energy stays constant for conv_iter iterations, then we say tha algorithm has converged.    
    """
    M=len(start_nodes)
    N=G.numberOfNodes()
    Gd=nk.graph.Graph(n=N, weighted=True, directed=True, edgesIndexed=False) #this graph is a directed version of G. It should not be subject to modifications throughout the dynamics.
    if G.isDirected():
        for u,v in G.iterEdges():
            Gd.addEdge(u,v)
            Gd.setWeight(u,v,1)
    else:
        for u,v in G.iterEdges():
            Gd.addEdge(u,v)
            Gd.setWeight(u,v,1)
            Gd.addEdge(v,u)
            Gd.setWeight(v,u,1)    
    
    paths = []
    G_traf=nk.graph.Graph(n=N, weighted=True, directed=False, edgesIndexed=False) #graph storing the the flow (or traffic) on its edges. 
    for u,v in G.iterEdges():
        G_traf.addEdge(u,v)
        G_traf.setWeight(u,v,0) #initialize all weights to zero (traffic is at zero in the beginning)

    # Initialize each path using Dijkstra
    for nu in range(M):
        dijkstra = nk.distance.Dijkstra(G, source=start_nodes[nu],target=end_nodes[nu])
        dijkstra.run()
        path = dijkstra.getPath(end_nodes[nu])
        paths.append(path.copy())   

        # total traffic on the graph
        for i in range(len(path)-1):
            G_traf.setWeight(path[i],path[i+1],G_traf.weight(path[i],path[i+1])+1)

    for u,v in Gd.iterEdges(): #initializing cost on all the graph: this way one will only have to modify the cost on the edges traversed by the considered path
        Gd.setWeight( u,v,  phi((G_traf.weight(u,v)+1))-phi(G_traf.weight(u,v))) #this is the increase in energy if one adds one path on the edge u,v.
     
    total_energy=sum([phi(w) for _,_,w in G_traf.iterEdgesWeights()]) #this is the objective function to minimize. When it becomes stationary, the algorithm stops. 
    count_conv=0
    np.random.seed(mcmc_seed)
    flag_conv=False
    for t,beta in enumerate(beta_schedule):
        if(not quiet):
            print(f"t={t}, beta={beta:.2e}, energy={total_energy}")
            
        for nu in range(M):
            path=paths[nu]
            # Remove the contribution of path nu
            for i in range(len(path)-1): 
                G_traf.setWeight(path[i],path[i+1],G_traf.weight(path[i],path[i+1])-1) # to get the traffic without path nu
                Gd.setWeight(path[i],path[i+1],phi((G_traf.weight(path[i],path[i+1])+1))- phi((G_traf.weight(path[i],path[i+1]))))
                Gd.setWeight(path[i+1],path[i],phi((G_traf.weight(path[i],path[i+1])+1))- phi((G_traf.weight(path[i],path[i+1]))))
            #Gd now contains the potential seen by path nu 

            new_path_list,_=saw.mcmcm_saw(beta, Gd, start_nodes[nu], end_nodes[nu], mcmc_steps, path) #compute update path using the mcmc. If None is passed as last argument instead of 'path', then the mcmc uses tha shortest path as initialization.
            new_path=new_path_list[-1] #take the last sample of the mcmc

            paths[nu]=new_path.copy()
            for i in range(len(new_path)-1): #add the contribution of the new path nu
                G_traf.setWeight(new_path[i],new_path[i+1],G_traf.weight(new_path[i],new_path[i+1])+1) #re adding the contribution of path nu
                Gd.setWeight(new_path[i],new_path[i+1], phi((G_traf.weight(new_path[i],new_path[i+1])+1))- phi((G_traf.weight(new_path[i],new_path[i+1])))) 
                Gd.setWeight(new_path[i+1],new_path[i], phi((G_traf.weight(new_path[i],new_path[i+1])+1))- phi((G_traf.weight(new_path[i],new_path[i+1])))) 
                
        #check for convergence in energy
        new_total_energy=sum([phi(w) for _,_,w in G_traf.iterEdgesWeights()])
        if(new_total_energy==total_energy):
            count_conv+=1
            if(count_conv>=conv_iters):
                flag_conv=True
                break
        else: 
            count_conv=0


        total_energy=new_total_energy
    return paths,flag_conv


