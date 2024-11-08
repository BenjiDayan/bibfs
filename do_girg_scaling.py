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
import uuid

# p = '../M3_ext_val_data/girgs3_scaling/'
# p = '../M3_ext_val_data/girgs_scaling_max_degrees_SQBfixed/'
p = '../M3_ext_val_data/girgs_scaling_no_early_stopping_25_09_2024/'
# fpaths = '../M3_ext_val_data/girgs3/'
fpaths = '../M3_ext_val_data/girgs3_mini/'


def read_girg(fn):
    fn_temp = str(uuid.uuid4())
    with open(fn, 'r') as f:
        lines = f.readlines()
        with open(fn_temp, 'w') as f2:
            f2.write('\n'.join(lines[2:]))
    g = nk.graphio.EdgeListReader(' ', 0).read(fn_temp)
    os.remove(fn_temp)
    return g

def process_graph(graph_name, g=None, **kwargs):
    if g is None:
        g = utils.graph_name_to_nk(graph_name)
    df = bbbfa.run_algos_on_g(g, 
        algos=bbbfa.ALGOS,
        # algos=[bbbfa.BiBFS_ExactExpandSmallerQueueBetter],
        record_max_degrees=True,
        early_stopping=False,                    
        **kwargs)
    df['graph'] = graph_name
    os.makedirs(p, exist_ok=True)
    output_path = p + graph_name + '.csv'
    df.to_csv(output_path, index=False)
    with open(p + graph_name + '_info.txt', 'w') as file:
        file.write(f'{g.numberOfNodes()} nodes, {g.numberOfEdges()} edges')


def process_graph2(fn):
    g = read_girg(fn)
    g = generators.get_largest_component(g, relabel=True)
    graph_name = fn.split('/')[-1][:-4]
    process_graph(graph_name, g, n_pairs=100)


if __name__ == '__main__':
    # graph_names = utils.input_names_real_with_cl
    # outnames = os.listdir(utils.p + '/real_fake_output_taufit')
    # outnames = [x[:-4] for x in outnames]
    # graph_names = set(utils.input_names_real_with_cl) - set(outnames)

    import glob
    args = glob.glob(fpaths + '*.txt')
    args = [x for x in args if 'dim=2' in x]
    # results = list(tqdm(map(lambda x: process_graph2(*x), args), total=len(args)))
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        results = list(tqdm(executor.map(process_graph2, args), total=len(args)))

    # graph_names = utils.input_names_cl
    # print(graph_names)
    # results = list(tqdm(map(process_graph, graph_names), total=len(graph_names)))
    # with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
    #     results = list(tqdm(executor.map(process_graph, graph_names), total=len(graph_names)))
    
    
    # for graph_name in graph_names:
    #     process_graph(graph_name, do_print=True)