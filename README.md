## Algorithms used in "Integer Traffic Assignment Problem: Algorithms and Insights on Random Graphs"
The integer traffic assignment problem (ITAP) is a routing problem consisting of finding the optimal paths with given endpoints, so to minimize the congestion on a graph.

Each folder is named after the respective algorithm and contains a file with a python implementation of tha algorithm as well as a jupyter notebook that provides an example of its usage.
1. RITAP is the relaxed ITAP solver, based on solving TAP (which is convex chen the nonlinearity in convex) and then projecting the solution back onto the integer constraints.
2. Greedy is a greedy algorithm that at each time step optimizes the position of one path keeping all the others fixed
3. simulated annealing is a simulated annealing based approach to ITAP. Similarly to greedy it updates one path at a time, but the update is stochastic.
4. saw sampler is an algorithm of independent interest that samples self avoiding walks with fixed endpoints on arbitrary graphs. It is also a component of the simulated annealing, since it is used to generate the stochastic paths.
    
