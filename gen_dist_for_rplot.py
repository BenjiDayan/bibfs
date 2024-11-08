import bbbfs_algorithms as bbbfa
import benji_utils as utils
import concurrent.futures
from tqdm import tqdm
import pandas as pd
import os

def process_graph(graph_name):
    g = utils.graph_name_to_nk(graph_name)
    df = bbbfa.run_algos_on_g(
        g, n_pairs=100,
        algos=bbbfa.ALGOS,
        # algos=[bbbfa.BiBFS_ExactExpandSmallerQueueBetter],
        record_max_degrees=True,
        early_stopping=False)
    # p = '../M3_ext_val_data/rplot_dist_queue_fixed_randomised_edges/'
    # p = '../M3_ext_val_data/real_graphs_max_degrees_QBfixed/'
    p = '../M3_ext_val_data/real_graphs_25_09_2024/'
    os.makedirs(p, exist_ok=True)
    output_path = p + graph_name + '.csv'
    df.to_csv(output_path, index=False)

if __name__ == '__main__':
    graph_names = utils.input_names_all
    # graph_names = utils.input_names_real
    print(graph_names)
    # results = list(tqdm(map(process_graph, graph_names), total=len(graph_names)))
    with concurrent.futures.ProcessPoolExecutor(max_workers=14) as executor:
        results = list(tqdm(executor.map(process_graph, graph_names), total=len(graph_names)))