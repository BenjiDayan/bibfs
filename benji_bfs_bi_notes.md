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


```bash
bash-5.1# ./code/release/dist_specific --s 9 --t 27 --seed 3404785993 --algo bfs_bi_balanced --no-header input_data/adj_array/DD_g714
bfs_bi_balanced,3404785993,9,27,5,0.010069,66
bash-5.1# ./code/release/dist_specific --s 9 --t 27 --seed 3404785993 --algo bfs_bi_node --no-header input_data/adj_array/DD_g714
bfs_bi_node,3404785993,9,27,6,0.011281,83
bash-5.1# ./code/release/dist_specific --s 9 --t 27 --seed 3404785993 --algo bfs_bi_node_exact --no-header input_data/adj_array/DD_g714
bfs_bi_node_exact,3404785993,9,27,6,0.012704,86
```


```bash
bash-5.1# ./code/release/dist_specific --s 13 --t 27 --seed 3404785993 --algo bfs_bi_balanced --no-header input_data/adj_array/DD_g714
bfs_bi_balanced,3404785993,13,27,4,0.009648,45
bash-5.1# ./code/release/dist_specific --s 13 --t 27 --seed 3404785993 --algo bfs_bi_node_exact --no-header input_data/adj_array/DD_g714
bfs_bi_node_exact,3404785993,13,27,5,0.012333,70
```

We fixed this but made a new bug wtf?

```bash
bash-5.1# ./code/release/dist_specific --s 9 --t 27 --seed 3404785993 --algo bfs_bi_node_exact --no-header input_data/adj_array/DD_g714
bfs_bi_node_exact,3404785993,9,27,3,0.011442,84
bash-5.1# ./code/release/dist_specific --s 9 --t 27 --seed 3404785993 --algo bfs_bi_node_balanced --no-header input_data/adj_array/DD_g714
--algo:  not in {bfs,bfs_bi_balanced,bfs_bi_always_swap,bfs_bi_node,bfs_bi_node_exact}
Run with --help for more information.
bash-5.1# ./code/release/dist_specific --s 9 --t 27 --seed 3404785993 --algo bfs_bi_balanced --no-header input_data/adj_array/DD_g714
bfs_bi_balanced,3404785993,9,27,5,0.010039,66
bash-5.1#
```