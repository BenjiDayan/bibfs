import bbbfs_algorithms as bbbfa
import benji_utils as utils
import concurrent.futures
from tqdm import tqdm
import pandas as pd
import os

def process_graph(graph_name):
    g = utils.graph_name_to_nk(graph_name)
    df = bbbfa.run_algos_on_g(g, n_pairs=100)
    p = '../M3_ext_val_data/rplot_dist/'
    os.makedirs(p, exist_ok=True)
    output_path = p + graph_name + '.csv'
    df.to_csv(output_path, index=False)

if __name__ == '__main__':
    graph_names = utils.input_names_all
    print(graph_names)
    # results = list(tqdm(map(process_graph, graph_names), total=len(graph_names)))
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        results = list(tqdm(executor.map(process_graph, graph_names), total=len(graph_names)))