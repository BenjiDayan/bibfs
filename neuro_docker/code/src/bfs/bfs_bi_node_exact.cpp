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
//    // Don't switch direction on new layer, rather on every new node
//    if (S->layer < S->dist[S->Q1.front()]) {
//        S->layer++;
//        update_cost_start_new_layer(*S);
//    }
    S = choose_direction(m_S_fward, m_S_bward);
    // Never zero out the cost - using Ulysse's total seen size.

//    std::cout << "costs: " <<m_S_fward.cost << " " << m_S_bward.cost << std::endl;


    //print the contents of the queue
//    std::cout << "Queue contents: ";
//    std::queue<node> temp = S->Q;
    //while (!temp.empty()) {
//    	std::cout << temp.front() << " ";
    //	temp.pop();
    //}

    node v = S->Q.front();
    S->Q.pop();

//    std::cout << "Popped: " << v << std::endl;


    for (node u : G.neighbors(v)) {
      m_search_space++;
      if (!S->node_found[u]) {
        // found a new node u
        S->Q_next.push(u);
        S->node_found[u] = true;
        S->dist[u] = S->dist[v] + 1;
        update_cost_node_found(*S, u, G);
        S->dirty_nodes.push_back(u);

//        std::cout<< "  Found: " << u << "; dist: " << S->dist[u] << std::endl;

        if (m_S_fward.node_found[u] && m_S_bward.node_found[u]) {
          // if found by both searches -> shortest path found
          // if first found then set as output_dist
          // if later found, take minimum
          unsigned new_dist = m_S_fward.dist[u] + m_S_bward.dist[u];
          output_dist = (found == false) ? new_dist : \
            std::min(output_dist, new_dist);
          found = true;
//          std::cout << "  Match found: " << u << "with dist: " << output_dist << std::endl;

          // !! Check if it really is the shortest path
          State* S2 = m_S_fward.Q.size() < m_S_bward.Q.size() ? &m_S_fward : &m_S_bward;
          while (!S2->Q.empty()) {
            node w = S2->Q.front();
            S2->Q.pop();
//            std::cout << "  Checking: " << w << "of dist: " << S2->dist[w] << std::endl;

            for (node u2 : G.neighbors(w)) {
//              std::cout << "    Checking neighbor: " << u2 << std::endl;
              if (!S2->node_found[u2]) {
                S2->node_found[u2] = true;
                S2->dist[u2] = S2->dist[w] + 1;
                S2->dirty_nodes.push_back(u2);
              }

              // Note this is debatable - it's a bit different from previous meaning.
              // Before: search space is how many nodes were ever added to a queue
              // Now: also add on the ones in the smallest queue that we double check
              m_search_space++;
              if (m_S_fward.node_found[u2] && m_S_bward.node_found[u2]) {
//                std::cout << "    Match found: " << u2 << "; fwd dist: " << m_S_fward.dist[u2] << "; bwd dist: " << m_S_bward.dist[u2] << std::endl;
                int alternative_output_dist = m_S_fward.dist[u2] + m_S_bward.dist[u2];
//                std::cout << "    Alternative match found: " << u2 << "with dist: " << alternative_output_dist << std::endl;
                if (alternative_output_dist < output_dist) {
                  return alternative_output_dist;
                  }
                }
            }
          }

//          return output_dist;
        }
        // We check if any other node u is found by both searches and shorter
        // Hence don't return output_dist in the if section, only after checking all other
        // nhbs of v.
      }
    }

//    std::cout << "found: " << found << std::endl;

    // If some node in u's nhbd made a path, we need to return.
    if (found) {
      return output_dist;
    }

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