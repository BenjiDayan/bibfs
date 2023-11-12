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




//#pragma once
//
//#include <memory>
//#include <queue>
//#include <vector>
//
//#include "framework/graph.hpp"
//
////#include "bfs/bfs_bi.hpp"
//
//class BFSBiNodeExact : public BFSBi {
// public:
//  BFSBiNodeExact(unsigned n) : BFSBi(n) {}
//  unsigned operator()(const Graph& G, node s, node t);
//
// protected:
//  struct State {
//    State() { /* default constructor implementation */ };
//    State(unsigned n);
//    void init(node v);
//    std::vector<unsigned> dist;
//    std::vector<bool> node_found;
//    std::queue<node> Q1;
//    std::queue<node> Q2;
//    unsigned layer;
//    unsigned cost;
//    std::vector<node> dirty_nodes;
//  };
//
//  State* choose_direction(State& S_fward, State& S_bward) const;
//
//  State m_S_fward;
//  State m_S_bward;
//
//  void update_cost_node_found(State& S, node v, const Graph& G) const;
//  void update_cost_start_new_layer(State& S) const;
//};
