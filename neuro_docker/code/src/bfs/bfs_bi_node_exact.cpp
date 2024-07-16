#include "bfs/bfs_bi_node_exact.hpp"

#include <limits>
#include <memory>
#include <iostream>

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

  bool found = false;
  unsigned output_dist = std::numeric_limits<unsigned>::max();

  while (!m_S_fward.Q.empty() && !m_S_bward.Q.empty()) {
    // Don't switch direction on new layer, rather on every new node
    S = choose_direction(m_S_fward, m_S_bward);
    // Never zero out the cost - using Ulysse's total seen size.

    //std::cout << "costs: " <<m_S_fward.cost << " " << m_S_bward.cost << std::endl;
    node v = S->Q.front();
    S->Q.pop();

    //std::cout << "Popped: " << v << std::endl;

    for (node u : G.neighbors(v)) {
      m_search_space++;
      if (!S->node_found[u]) {
        // found a new node u
        S->Q_next.push(u);
        S->node_found[u] = true;
        S->dist[u] = S->dist[v] + 1;
        update_cost_node_found(*S, u, G);
        S->dirty_nodes.push_back(u);

        //std::cout<< "  Found: " << u << "; dist: " << S->dist[u] << std::endl;


        if (m_S_fward.node_found[u] && m_S_bward.node_found[u]) {
          // if found by both searches -> potential shortest path found
          // Could be that v ~ u1, u2 with dist of u1 longer than dist of u2, so we have to check all nhbs of
          // v to ensure that we haven't missed an even shorter path
          unsigned new_dist = m_S_fward.dist[u] + m_S_bward.dist[u];
          //std::cout << "  Match found: " << u << "with dist: " << new_dist << std::endl;
          if (found == false) {
            output_dist = new_dist;
            found = true;
          }
          else {
            if (new_dist < output_dist) {
              return new_dist;
            }
          }

        }
      }
    }

    // If we did find a path, we need to check if there is a shorter one:
    // Expand all nodes in the smallest queue
    if (found == true) {
      State* S2 = m_S_fward.Q.size() < m_S_bward.Q.size() ? &m_S_fward : &m_S_bward;
      while (!S2->Q.empty()) {
        node w = S2->Q.front();
        S2->Q.pop();
        //std::cout << "  Checking: " << w << "of dist: " << S2->dist[w] << std::endl;

        for (node u2 : G.neighbors(w)) {
          //std::cout << "    Checking neighbor: " << u2 << std::endl;
          if (!S2->node_found[u2]) {
            S2->node_found[u2] = true;
            S2->dist[u2] = S2->dist[w] + 1;
            S2->dirty_nodes.push_back(u2);
          }
          // Note this is debatable - it's a bit different from previous meaning.
          // Before: search space is how many nodes were ever added to a queue
          // Now: also add on extra uncovered nodes as we check them.
          m_search_space++;
          if (m_S_fward.node_found[u2] && m_S_bward.node_found[u2]) {
            //std::cout << "    Match found: " << u2 << "; fwd dist: " << m_S_fward.dist[u2] << "; bwd dist: " << m_S_bward.dist[u2] << std::endl;
            int alternative_output_dist = m_S_fward.dist[u2] + m_S_bward.dist[u2];
            //std::cout << "    Alternative match found: " << u2 << "with dist: " << alternative_output_dist << std::endl;
            if (alternative_output_dist < output_dist) {
              return alternative_output_dist;
            }
          }
        }
      }
     }

    // If we never found a smaller path, after finishing expanding both the initial v's nhbs,
    // and then the whole of the shorter queue, well then return what we've got.
    if (found) {
      return output_dist;
    }

    // If an inner layer queue is exhausted, replace it with the next queue that has been prepared.
    if (S->Q.empty()) {
      std::swap(S->Q, S->Q_next);
      S->layer++;
      update_cost_start_new_layer(*S);
    }
  }

  return std::numeric_limits<unsigned>::max();
}

void BFSBiNodeExact::update_cost_node_found(State& S, node v,
                                               const Graph& G) const {
  // This is an approximation!
  // "The cost of exploring a layer is estimated by the sum of vertex degrees in that layer"
  // (from the paper - on balanced layerwise)
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