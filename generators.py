import math
import random
import networkit as nk
import numpy as np
from networkit.graph import Graph

import logging
import networkx as nx

import os
import powerlaw
import sys
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def powerlaw_fit_graph(g) -> float:
    dd = sorted(nk.centrality.DegreeCentrality(g).run().scores(), reverse=True)
    with HiddenPrints():
        fit = powerlaw.Fit(dd, discrete=True)
    return fit.power_law.alpha

def powerlaw_dist(tau=2.5, x_min=1, n=1000, discrete=True) -> np.ndarray:
    # """sample from a tau exponent power law distribution
    # pdf: prop to x^-(a+1), i.e. tau = a+1
    # mean: ((tau-1) x_min)/(tau - 2) for tau > 2
    # x_min: support is [x_min, inf]
    # size: number of samples to draw
    # """
    # a = tau-1
    # pareto = (np.random.pareto(a, size=n) + 1) * x_min
    # return pareto
    
    # Actually I think the power_law alpha fit is the x^{-tau} exponeent of the pdf.
    dist= powerlaw.Power_Law(xmin=x_min, parameters=[tau], discrete=discrete)
    out = dist.generate_random(n)
    if discrete:
        return out.astype(np.int32)
    return out


def get_largest_component(g: Graph, relabel=True) -> Graph:
    cc = nk.components.ConnectedComponents(g)
    cc.run()
    comp_idx_size_pairs = list(cc.getComponentSizes().items())
    comp_idx_size_pairs.sort(key=lambda x: x[1], reverse=True)
    nodes = cc.getComponents()[comp_idx_size_pairs[0][0]]
    g_restricted = nk.graphtools.subgraphFromNodes(g, nodes)
    if not relabel:
        return g_restricted

    return relabel_graph(g_restricted)

def relabel_graph(g: Graph) -> Graph:
    g_nx = nk.nxadapter.nk2nx(g)
    g_nx_relabelled = nx.relabel.convert_node_labels_to_integers(g_nx, first_label=0, ordering='default')
    return nk.nxadapter.nx2nk(g_nx_relabelled)

def fit_connected_chunglu_to_g(g, tau=None, iters=5, tol=0.1):
    """Tries to make a ChungLu graph with the same number of nodes and edges as g.
    We do weights from tau powerlaw dist. But we have to increase the number of nodes to ensure connectedness.

    How do we simultaneously hit the node and edge count???

    Good question. We can't. So we'll just have to do our best.
    """

    n, e = g.numberOfNodes(), g.numberOfEdges()
    if tau is None:
        tau = powerlaw_fit_graph(g)
    n2, e2 = n, e
    for _ in range(iters):
        g = generate_chung_lu(n2, tau=tau, desiredAvgDegree=2*e2/n2)
        g_restrict = get_largest_component(g, relabel=False)
        n_act, e_act = g_restrict.numberOfNodes(), g_restrict.numberOfEdges()
        print(f"n2={n2}, e2={e2}, n_act={n_act}, e_act={e_act}")

        if np.abs((n-n_act)/n) < tol and np.abs((e-e_act)/e) < tol:
            return relabel_graph(g_restrict), True

        n2 *= n/n_act
        n2 = min(int(n2), 2*n)  # don't blow it up??
        e2 *= (e/n) / (e_act/n_act)  # adjust e2 by the desired avg degree / actual avg degree
    
    return g_restrict, False


def generate_chung_lu(n, tau=2.5, desiredAvgDegree=50.0):
    weights = powerlaw_dist(tau, 1, n, discrete=False)#
    # weights = weights / np.sqrt(weights.sum())

    # # fit c to get the desired avg degree.
    # c, g_cl = chung_lu_fit_c_lite(desiredAvgDegree, weights)
    # return g_cl

    # This one doesn't want normalised weights
    c, g_cl = chung_lu_fit_c_lite2(desiredAvgDegree, weights)
    return g_cl

# We want a version of 

def chung_lu_fit_c(g, weights):
    """p_uv = min(1, c wu wv)
    NB we assume that weights are normalised so actually wu = w'u/sqrt(W)
    so no need to normalise.

    This is just a scale factor though, so it would merely affect the value of c.

    we have to fit c so that sum p_uv is about the degree of g.


    e.g. weights could be the degree sequence

    """
    if type(g) is nk.Graph:
        num_edges = g.numberOfEdges()  # = (1/2) Sum p_uv
    else:  # hack if you want to use g as e.g. 50.0 desired average degree.
        desired_avg_degree = g
        num_edges = desired_avg_degree * len(weights) / 2
    c = 1.0
    for _ in range(10):
        probs = np.minimum(c * np.outer(weights, weights), 1)
        E_edges = probs.sum() - np.diag(probs).sum()
        E_edges = E_edges / 2
        c = c * (num_edges / E_edges)

    probs = np.minimum(c * np.outer(weights, weights), 1)

    return c, probs


def chung_lu_fit_c_lite(g, weights):
    """p_uv = min(1, c wu wv)
    NB we assume that weights are normalised so actually wu = w'u/sqrt(W)
    so no need to normalise.

    This is just a scale factor though, so it would merely affect the value of c.

    we have to fit c so that sum p_uv is about the degree of g.


    e.g. weights could be the degree sequence

    """
    if type(g) is nk.Graph:
        num_edges = g.numberOfEdges()  # = (1/2) Sum p_uv
    else:  # hack if you want to use g as e.g. 50.0 desired average degree.
        desired_avg_degree = g
        num_edges = desired_avg_degree * len(weights) / 2


    c = 1.0
    for _ in range(10):
        E_edges = chung_lu_generate(weights, c, graph=False)
        c = c * (num_edges / E_edges)

    g_out = chung_lu_generate(weights, c, graph=True)
    return c, g_out


def discretise_weights(weights):
    """Discretise weights to integers"""
    out = []
    randoms = np.random.rand(len(weights))
    floored_weights = np.floor(weights)
    out = floored_weights.astype(np.int32)
    out += (randoms < weights - floored_weights).astype(np.int32)
    return out

    # for w in weights:
    #     out.append(math.floor(w))
    #     if np.random.rand() < w - out[-1]:
    #         out[-1] += 1
    # return out

def chung_lu_fit_c_lite2(g, weights: np.ndarray, iters=5, do_print=False):
    E_edges = weights.sum() / 2
    if type(g) is nk.Graph:
        num_edges = g.numberOfEdges()  # = (1/2) Sum p_uv
    else:  # hack if you want to use g as e.g. 50.0 desired average degree.
        desired_avg_degree = g
        num_edges = desired_avg_degree * len(weights) / 2

    c = num_edges / E_edges

    # weights_disc_scaled = discretise_weights(weights * num_edges / E_edges)
    # g_out = nk.generators.ChungLuGenerator(weights_disc_scaled).generate()

    for _ in range(iters):
        weights_disc_scaled = discretise_weights(weights * c)
        g_out = nk.generators.ChungLuGenerator(weights_disc_scaled).generate()
        E_edges = g_out.numberOfEdges()
        if do_print:
            print(f"num_edges={num_edges}, E_edges={E_edges}, c={c}")
        c = c * (num_edges / E_edges)

    return c, g_out

    
    

# This is a less ram hungry version yay.
def chung_lu_generate(weights, c, graph=False):
    if graph:
        g_out = nk.Graph(len(weights))
    else:
        edge_probs = 0
    for i in range(1, len(weights)):
        # i.e. 1 to 1+i, 2 to 2 + i, ..., n-i to n
        probs_k_to_kpi = np.minimum(c * weights[:-i] * np.roll(weights, -i)[:-i], 1)
        if graph:
            ps = np.random.uniform(size=len(probs_k_to_kpi))
            for k in np.argwhere(ps < probs_k_to_kpi).flatten():
                g_out.addEdge(k, k+i)
        else:
            edge_probs += probs_k_to_kpi.sum()
    
    return g_out if graph else edge_probs



def upper_triangular_to_edgelist_fastest(mat: np.ndarray) -> list[tuple[int, int]]:
    """Converts an upper triangular matrix to an edgelist, fastest possible"""
    return list(zip(*np.nonzero(mat)))

def edge_list_to_nk_graph(edge_list: list[tuple[int, int]], n) -> Graph:
    g = nk.Graph(n)
    for u, v in edge_list:
        g.addEdge(u, v)

    return g