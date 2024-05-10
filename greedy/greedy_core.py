import numpy as np
import networkit as nk
import copy

def greedy_optimizer(G,start_nodes,end_nodes,phi,max_steps=20,quiet=True, init_paths=None):
    """
    This function implements the greedy algorithm for the integer optimal routing problem.

    At each iteration, the algorithm loops over origin-destination pairs and recomputes the shortest path for each pair. 
    The shortest path is computed in a weighted graph where each path sees an effective cost created by all other paths.

    Args:
    G: networkit.graph (directed or undirected)
    start_nodes: list of integers
    end_nodes: list of integers
    phi: function, nonlinearity in the Hamiltonian
    max_steps: int, maximum number of iterations before halting
    quiet: bool, if True, the function does not print anything
    init_paths: list of lists of integers, initial paths to start the algorithm from. If None, the initial paths are computed using Dijkstra's algorithm.

    Returns:
    paths: list of lists of integers, the optimal paths found by the algorithm
    flag_conv: bool, True if the algorithm converged before reaching max_steps iterations, False otherwise

    """
    
    M=len(start_nodes)
    N=G.numberOfNodes()
    flag_conv=False
    delta_phi=lambda x: phi(x+1)-phi(x) #this is the increase in the cost function when the traffic on an edge increases by one unit
    G_traf=nk.graph.Graph(n=N, weighted=True, directed=False, edgesIndexed=False)
    for u,v in G.iterEdges():
        G_traf.addEdge(u,v)
        G_traf.setWeight(u,v,0) #initialize all weights to zero (traffic is at zero in the beginning)
              
    if(init_paths==None):
        paths=[]
        # Initialize each path using Dijkstra
        for nu in range(M):
            dijkstra = nk.distance.Dijkstra(G, source=start_nodes[nu],target=end_nodes[nu])
            dijkstra.run()
            path = dijkstra.getPath(end_nodes[nu])
            paths.append(path.copy())
            
    else:
        paths = copy.deepcopy(init_paths)

        # total traffic on the graph
    for nu in range(M):
        path=paths[nu]
        for i in range(len(path)-1):
            G_traf.setWeight(path[i],path[i+1],G_traf.weight(path[i],path[i+1])+1)
            
    for u,v in G.iterEdges(): #initializing cost on all the graph: this way one will only have to modify the cost on the edges traversed by the considered path
        G.setWeight(u,v, delta_phi(G_traf.weight(u,v)))

    total_energy=sum([phi(w) for _,_,w in G_traf.iterEdgesWeights()])#this is the objective function to minimize. When it becomes stationary, the algorithm stops. 
    for t in range(max_steps):
        if(not quiet):
            print(f"t={t} energy={total_energy}")
            
        for nu in range(M):
            path=paths[nu]
            # Remove the contribution of path nu
            for i in range(len(path)-1): 
                G_traf.setWeight(path[i],path[i+1],G_traf.weight(path[i],path[i+1])-1) # to get the traffic without path nu
                G.setWeight(path[i],path[i+1],delta_phi(G_traf.weight(path[i],path[i+1])))
            #G now contains the potential seen by path nu 
            
            # Compute the new optimal path nu
            dijkstra = nk.distance.Dijkstra(G, source=start_nodes[nu],target=end_nodes[nu]) 
            dijkstra.run()
            new_path=dijkstra.getPath(end_nodes[nu])
            
            paths[nu]=new_path.copy()
            for i in range(len(new_path)-1): #add the contribution of the new path nu
                G_traf.setWeight(new_path[i],new_path[i+1],G_traf.weight(new_path[i],new_path[i+1])+1) #re adding the contribution of path nu
                G.setWeight(new_path[i],new_path[i+1], delta_phi(G_traf.weight(new_path[i],new_path[i+1])))
        #check for convergence in energy
        new_total_energy=sum([phi(w) for _,_,w in G_traf.iterEdgesWeights()])
        if(new_total_energy==total_energy):
            flag_conv=True
            break
        total_energy=new_total_energy
        
    for u,v in G.iterEdges(): #restore graph G to its previous form
        G.setWeight(u,v,1)
    return paths,flag_conv