#pragma once

#include "bfs/bfs_bi.hpp"

class BFSBiNodeBalanced : public BFSBi {
 public:
  BFSBiNodeBalanced(unsigned n) : BFSBi(n) {}

 protected:
  void update_cost_node_found(State& S, node v, const Graph& G) const;
  void update_cost_start_new_layer(State& S) const;
};
