#include "bfs/bfs_bi_node_balanced.hpp"

void BFSBiNodeBalanced::update_cost_node_found(State& S, node v,
                                               const Graph& G) const {
  S.cost += G.degree(v);
}

void BFSBiNodeBalanced::update_cost_start_new_layer(State& S) const {
  S.cost = 0;
}
