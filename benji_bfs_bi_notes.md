### bfs_bi.hpp runthrough
- takes in G, s, t

#### struct State
We have an m_S_fward and m_S_bward version of this

- dist, node_found: vectors
- queue Q
- layer, cost


bfs_bi.hpp defines a virtual method choose_direction which is implemented in bfs_bi.cpp, and two **pure** virtual funcs update_cost_node_found, update_cost_start_new_layer (because = 0 after) which have to be defined in subcalsses bfs_bi_balanced.hpp


How does bfs_bi.cpp work?

note bfs_bi is never meant to be used itself, only subclasses thereof, as bfs_bi doesn't implement its own pure virtual methods update_cost_node_found etc.



- m_S_fward, m_S_bward initted

- We shoose S to be one of these fward/bward.
- S.layer is the current layer/dist at the frontier
- if we finish up the current layer, we do S = choose_direction(fwd, bwd). This compares the costs of fwd and bwd and picks the lower cost one.
	- currently this actually then resets the cost to 0 of our newly picked queue.

- pop v from front of queue.
	- for each u ~ v, if is new for S, add to S.node_found; update_cost_node_found(S, u, G)

	- if fwd.node_found[u] and bwd.node_found[u]: return!


So as it is, we choose_direction S = fwd or bwd at each layer exhaustion, which we do by taking minimal of fwd.cost, bwd.cost

- bfs_bi_balanced has S.cost += G.degree(u) when u is first found. It actually also resets at each layer.
- bfs_bi_always_swap does nothing when u are found, rather just does S.cost++ when update_cost_start_new_layer


Ahhh so One BFSBi object is instantiated, and possibly used multiple times. Maybe multiples times just for the same graph? .dirty_nodes is used to clean up after each operation. This really just cleans everything up.



