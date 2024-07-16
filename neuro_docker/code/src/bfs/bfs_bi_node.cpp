#include "bfs/bfs_bi_node.hpp"

#include <limits>
#include <memory>

unsigned BFSBiNode::operator()(const Graph& G, node s, node t) {
  m_search_space = 0;
  if (s == t) return 0;

  m_S_fward.init(s);
  m_S_bward.init(t);

  update_cost_node_found(m_S_fward, s, G);
  update_cost_node_found(m_S_bward, t, G);

  State* S = choose_direction(m_S_fward, m_S_bward);
  update_cost_start_new_layer(*S);

  while (!m_S_fward.Q.empty() && !m_S_bward.Q.empty()) {
    // Don't switch direction on new layer, rather on every new node
    if (S->layer < S->dist[S->Q.front()]) {
        S->layer++;
        update_cost_start_new_layer(*S);
    }
    S = choose_direction(m_S_fward, m_S_bward);
    // Never zero out the cost - using Ulysse's total seen size.

    node v = S->Q.front();
    S->Q.pop();
    for (node u : G.neighbors(v)) {
      m_search_space++;
      if (!S->node_found[u]) {
        // found a new node u
        S->Q.push(u);
        S->node_found[u] = true;
        S->dist[u] = S->dist[v] + 1;
        update_cost_node_found(*S, u, G);
        S->dirty_nodes.push_back(u);

        if (m_S_fward.node_found[u] && m_S_bward.node_found[u]) {
          // if found by both searches -> shortest path found

          return m_S_fward.dist[u] + m_S_bward.dist[u];
        }
      }
    }
  }

  return std::numeric_limits<unsigned>::max();
}

void BFSBiNode::update_cost_node_found(State& S, node v,
                                               const Graph& G) const {
  S.cost += G.degree(v);
}

void BFSBiNode::update_cost_start_new_layer(State& S) const {
//  S.cost = 0;
}


//unsigned BFSBiNode::search_space() const { return m_search_space; }
//
//BFSBiNode::State::State(unsigned n) : dist(n), node_found(n, false) {}
//
//void BFSBiNode::State::init(node v) {
//  // tidy up previous search
//  for (node u : dirty_nodes) {
//    node_found[u] = false;
//  }
//  dirty_nodes.clear();
//
//  // init the new search
//  Q = std::queue<node>();
//  layer = 0;
//  cost = 0;
//  Q.push(v);
//  node_found[v] = true;
//  dist[v] = 0;
//  dirty_nodes.push_back(v);
//}
//
//
//BFSBiNode::State* BFSBiNode::choose_direction(State& S_fward,
//                                              State& S_bward) const {
//  return S_fward.cost < S_bward.cost ? &S_fward : &S_bward;
//}
