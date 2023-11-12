#include "bfs/bfs_bi_node_exact.hpp"

#include <limits>
#include <memory>

//BFSBiNodeExact::BFSBiNodeExact(unsigned n) : m_S_fward(n), m_S_bward(n) {}

unsigned BFSBiNodeExact::operator()(const Graph& G, node s, node t) {
  m_search_space = 0;
  if (s == t) return 0;

  m_S_fward.init(s);
  m_S_bward.init(t);

  update_cost_node_found(m_S_fward, s, G);
  update_cost_node_found(m_S_bward, t, G);

  State* S = choose_direction(m_S_fward, m_S_bward);
  update_cost_start_new_layer(*S);

  while (!m_S_fward.Q.empty() && !m_S_bward.Q.empty()
         && !m_S_fward.Q_next.empty() && !m_S_bward.Q_next.empty()) {
    // Don't switch direction on new layer, rather on every new node
//    if (S->layer < S->dist[S->Q1.front()]) {
//        S->layer++;
//        update_cost_start_new_layer(*S);
//    }
    if (S->Q.empty()) {
      std::swap(S->Q, S->Q_next);
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
        S->Q_next.push(u);
        S->node_found[u] = true;
        S->dist[u] = S->dist[v] + 1;
        update_cost_node_found(*S, u, G);
        S->dirty_nodes.push_back(u);

        if (m_S_fward.node_found[u] && m_S_bward.node_found[u]) {
          // if found by both searches -> shortest path found
          int output_dist = m_S_fward.dist[u] + m_S_bward.dist[u];

          // !! Check if it really is the shortest path
          State* S2 = m_S_fward.Q.size() < m_S_bward.Q.size() ? &m_S_fward : &m_S_bward;
          while (!S2->Q.empty()) {
            node w = S2->Q.front();
            S2->Q.pop();
            // Note this is debatable - it's a bit different from previous meaning.
            // Before: search space is how many nodes were ever added to a queue
            // Now: also add on the ones in the smallest queue that we double check
            m_search_space++;
            if (m_S_fward.node_found[w] && m_S_bward.node_found[w]) {
              int alternative_output_dist = m_S_fward.dist[w] + m_S_bward.dist[w];
              if (alternative_output_dist < output_dist) {
                return alternative_output_dist;
              }
            }
          }

          return output_dist;
        }
      }
    }
  }

  return std::numeric_limits<unsigned>::max();
}

void BFSBiNodeExact::update_cost_node_found(State& S, node v,
                                               const Graph& G) const {
  S.cost += G.degree(v);
}

void BFSBiNodeExact::update_cost_start_new_layer(State& S) const {
//  S.cost = 0;
}


//void BFSBiNodeExact::State::init(node v) {
//  // tidy up previous search
//  for (node u : dirty_nodes) {
//    node_found[u] = false;
//  }
//  dirty_nodes.clear();
//
//  // init the new search
//  Q = std::queue<node>();
//  Q_next = std::queue<node>();
//  layer = 0;
//  cost = 0;
//  Q1.push(v);
//  node_found[v] = true;
//  dist[v] = 0;
//  dirty_nodes.push_back(v);
//}