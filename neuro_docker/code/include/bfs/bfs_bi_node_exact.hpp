#pragma once

#include "bfs/bfs_bi.hpp"

class BFSBiNodeExact : public BFSBi {
 public:
  BFSBiNodeExact(unsigned n) : BFSBi(n) {}
  unsigned operator()(const Graph& G, node s, node t);

 protected:
  void update_cost_node_found(State& S, node v, const Graph& G) const;
  void update_cost_start_new_layer(State& S) const;
};
