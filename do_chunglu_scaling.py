import bbbfs_algorithms as bbbfa
import benji_utils as utils
import concurrent.futures
from tqdm import tqdm
import pandas as pd
import os
import networkit as nk
import numpy as np
import random
import generators
from bbbfs_algorithms import BiBFS_LayerbalancedFull, BiBFS_ExactExpandSmallerQueueBetter

p = '../M3_ext_val_data/chunglu_scaling_LBF_VBSQB/'

def process_graph(graph_name, g=None, **kwargs):
    if g is None:
        g = utils.graph_name_to_nk(graph_name)
    df = bbbfa.run_algo                                       s_on_g(g, 
        algos=[BiBFS_LayerbalancedFull, BiBFS_ExactExpandSmallerQueueBetter],                      
        **kwargs)
    df['graph'] = graph_name
    os.makedirs(p, exist_ok=True)
    output_path = p + graph_name + '.csv'
    df.to_csv(output_path, index=False)
    with open(p + graph_name + '_info.txt', 'w') as file:
        file.write(f'{g.numberOfNodes()} nodes, {g.numberOfEdges()} edges')

def process_graph2(n, deg, ple, seed, **kwargs):
    graph_name = f'cl_n={n}_deg={deg}_ple={ple}_seed={seed}'
    nk.setSeed(seed, True)
    np.random.seed(seed)
    random.seed(seed)
    g = generators.generate_chung_lu(n, tau=ple, desiredAvgDegree=deg)
    g = generators.get_largest_component(g, relabel=True)

    process_graph(graph_name, g, n_pairs=100, **kwargs)

if __name__ == '__main__':
    # graph_names = utils.input_names_real_with_cl
    # outnames = os.listdir(utils.p + '/real_fake_output_taufit')
    # outnames = [x[:-4] for x in outnames]
    # graph_names = set(utils.input_names_real_with_cl) - set(outnames)

    ples = [2.0, 2.1, 2.3, 2.5, 2.7, 2.9, 3.0, 3.3, 4.0, 6.0, 9.0, 13.0, 18.0, 25.0]
    ns = [100, 150, 300, 500, 750, 1000, 1500, 3000, 5000, 7500, 10000, 15000, 30000, 50000, 80000]
    # ns = [1000, 1500]
    degs = [10, 30, 60]
    seeds = [1,2,3,4,5]

    import itertools
    args = list(itertools.product(ns, degs, ples, seeds))
    # results = list(tqdm(map(lambda x: process_graph2(*x), args), total=len(args)))
    with concurrent.futures.ProcessPoolExecutor(max_workers=14) as executor:
        results = list(tqdm(executor.map(process_graph2, *zip(*args)), total=len(args)))

    # graph_names = utils.input_names_cl
    # print(graph_names)
    # results = list(tqdm(map(process_graph, graph_names), total=len(graph_names)))
    # with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
    #     results = list(tqdm(executor.map(process_graph, graph_names), total=len(graph_names)))
    
    
    # for graph_name in graph_names:
    #     process_graph(graph_name, do_print=True)