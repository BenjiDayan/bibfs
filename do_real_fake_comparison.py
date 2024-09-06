import bbbfs_algorithms as bbbfa
import benji_utils as utils
import concurrent.futures
from tqdm import tqdm
import pandas as pd
import os

def process_graph_blasius(graph_name):
    df = bbbfa.do_real_fake_comparison(graph_name, n_pairs=5000, algos=[bbbfa.BiBFS_Layerbalanced])
    p = '../M3_ext_val_data/real_fake_output_blasius/'
    os.makedirs(p, exist_ok=True)
    output_path = p + graph_name + '.csv'
    df.to_csv(output_path, index=False)

def process_graph(graph_name):
    df = bbbfa.do_real_fake_comparison(graph_name, n_pairs=1000)
    p = '../M3_ext_val_data/real_fake_output_taufit/'
    os.makedirs(p, exist_ok=True)
    output_path = p + graph_name + '.csv'
    df.to_csv(output_path, index=False)

if __name__ == '__main__':
    # graph_names = utils.input_names_real_with_cl
    # outnames = os.listdir(utils.p + '/real_fake_output_taufit')
    # outnames = [x[:-4] for x in outnames]
    # graph_names = set(utils.input_names_real_with_cl) - set(outnames)
    graph_names = utils.input_names_real_with_cl
    print(graph_names)
    # results = list(tqdm(map(process_graph, graph_names), total=len(graph_names)))
    with concurrent.futures.ProcessPoolExecutor(max_workers=14) as executor:
        results = list(tqdm(executor.map(process_graph, graph_names), total=len(graph_names)))