#pragma once

#include "bfs/bfs_bi_node.hpp"

class BFSBiNodeBalanced : public BFSBiNode {
 public:
  BFSBiNodeBalanced(unsigned n) : BFSBiNode(n) {}

 protected:
  void update_cost_node_found(State& S, node v, const Graph& G) const;
  void update_cost_start_new_layer(State& S) const;
};
