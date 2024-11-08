import pandas as pd
import os
import generators
import matplotlib.pyplot as plt
import numpy as np
import networkit as nk
from networkit import Graph
import powerlaw

# df = pd.read_csv('../dist.csv')

p = '../M3_ext_val_data/'
graph_names = os.listdir(p + 'edge_list')
graph_names.sort()

df_stats = pd.read_csv(p + 'output_data/overall_df_stats.csv')

df_tbl = pd.read_csv('../blasius_tbl.csv')
del df_tbl['Unnamed: 0']
input_names_real_filtered = df_tbl.loc[df_tbl['type'] == 'real']['graph'].tolist()  # This is the 2740 blasius subset

import networkit as nk

# g = nk.Graph(p + 'edge_list/DD_g63')
import glob
edgelists = glob.glob(p + 'edge_list/*')
edgelists[:10]

edgelists_cl = glob.glob(p + 'edge_list_cl_taufit/*')

def graph_name_to_nk(graph_name, prefix='edge_list', cl=False):
    """Loads a graph by name"""
    # NB about 7% of real graphs aren't in the cl folder - their
    # shape didn't fit.
    if cl:
        prefix = 'edge_list_cl_taufit'
    fn = f'{p}{prefix}/{graph_name if not cl else graph_name + "_cl"}'
    assert(fn in edgelists if not cl else fn in edgelists_cl)
    # fn = edgelists[edgelists.index(f'{p}edge_list/{graph_name}')]
    return nk.graphio.EdgeListReader(' ', 0).read(fn)

def graph_name_to_resultsdf(graph_name):
    algo_name_map = {
        'python-BiBFS_Layerbalanced': 'LB',
        'python-BiBFS_VertexBalancedApproximate': 'VBA',
        'python-BiBFS_ExactExpandSmallerQueue': 'VBSQ',
        'python-BiBFS_ExactCheckDirectEdges': 'VBDE',
        'python-BiBFS_EdgeBalancedApproximate': 'EBA',
    }

    fn = f'{p}real_fake_output_taufit/{graph_name}.csv'
    df = pd.read_csv(fn)
    shape_per_graph_type = df.shape[0] // (2*len(algo_name_map))
    j = list(df.columns).index('graph')
    for i in range(len(algo_name_map)):
        df.iloc[(2*i +1) * shape_per_graph_type: (2*i+2) * shape_per_graph_type, j] = f'{graph_name}_cl'
    
    
    # rename df.algo to algo_name_map[df.algo]
    df.algo = df.algo.map(algo_name_map)
    return df

def check_cls(num=20):
    """temp func just to check that CL graphs were generated successfully."""
    for graph_name in input_names_real[:num]:
        g = graph_name_to_nk(graph_name)
        try:
            g_cl = graph_name_to_nk(graph_name, cl=True)
        except:
            print(f'No cl graph for {graph_name}')
            continue
        print(graph_name, g.numberOfNodes(), g_cl.numberOfNodes(), '     ', g.numberOfEdges(), g_cl.numberOfEdges())

# def real_graph_to_gen_cl(graph_name, tau=2.5):
#     g = graph_name_to_nk(graph_name)
#     g_cl = generators.fit_connected_chunglu_to_g(g, tau=tau, iters=3, tol=0.1)
#     out_fp = f'{p}edge_list/{graph_name}_cl_tau{tau}'
#     nk.graphio.EdgeListWriter(' ', 0).write(g_cl, out_fp)

def slice_df(df, s, t, name):
    return df.loc[(df.s==s) & (df.t == t) & (df.graph == name)]


input_names_all = graph_names
input_names_all.sort()

input_names_girg_deg_scaling = [
    name for name in input_names_all if name.startswith("girg_deg_scaling_")
]
# special graphs: high degree
input_names_gen_high_deg = [
    name
    for name in input_names_all
    if ("deg=20" in name or "m=500000" in name)
    and not name.startswith("girg_deg_scaling_")
]
# main data set
input_names = [
    name
    for name in input_names_all
    if not name.startswith("girg_deg_scaling_")
    and "deg=20" not in name
    and "m=500000" not in name
]
# girg part of main data set
input_names_girg = [
    name for name in input_names if name.startswith("girg_")
]
input_names_cl = [
    name for name in input_names if name.startswith("cl_")
]
input_names_er = [
    name for name in input_names if name.startswith("er_")
]

# all other graphs in input_names - real networks
input_names_real = list(set(input_names) - set(input_names_girg) - set(input_names_cl) - set(input_names_er))
input_names_real.sort()


# fake graph counterparts for real graphs
cl_fake_graphs = os.listdir(p + 'edge_list_cl_taufit')

input_names_real_with_cl = [name for name in input_names_real if name + '_cl' in cl_fake_graphs]


def plot_degree_dist(g: Graph, pl_fit=False, vlines=0):
    if type(g) is nk.Graph:
        dd = sorted(nk.centrality.DegreeCentrality(g).run().scores(), reverse=True)
    elif type(g) is np.ndarray and np.issubdtype(g.dtype, np.integer):
        dd = sorted(g.astype(np.int64), reverse=True)
    else:
        raise Exception('g should be an nk Graph, or a np.ndarray of integers >=1')
    degrees, numberOfNodes = np.unique(dd, return_counts=True)
    # fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    plt.subplot(121)
    # plt.sca(axes[0])
    plt.xscale("log")
    plt.xlabel("degree")
    plt.yscale("log")
    plt.ylabel("number of nodes")
    # plt.scatter(degrees, numberOfNodes, s=1.1, marker='x')
    plt.scatter(degrees, numberOfNodes)
    if pl_fit:
        fit = powerlaw.Fit(dd, discrete=True)
        print(f'powerlaw alpha: {fit.power_law.alpha:.3f}')
#         fit.power_law.plot_pdf(linestyle='--', color='purple')
        plt.axvline(fit.xmin, linestyle='--', color='r', label=f'xmin: {fit.xmin}')
#         plt.axvline(fit.xmax, linestyle='--', color='pink', label=f'xmax: {fit.xmax}')
        y = fit.power_law.pdf()
        plt.plot(fit.data, y * len(fit.data), linestyle='--', color='purple')
        
    if vlines > 0:  # plot like quartile lines for number of nodes.
        # rough q-tiles
        q = vlines
        colors = plt.cm.rainbow(np.linspace(0, 1, q))
        rev_dd = list(reversed(dd))
        for i in range(1, q):
            plt.axvline(rev_dd[i * len(dd)//q], label=f'qtile-{i}/{q}', c=colors[i])
        plt.legend()
    # plt.show()
    plt.subplot(122)

    one_minus_cdf = 1. * np.arange(len(dd)) / (len(dd) - 1)
    plt.xscale("log")
    plt.xlabel("degree")
    plt.yscale("log")
    plt.ylabel("1 - CDF")
    plt.plot(dd, one_minus_cdf)
    ax = plt.gca()
    if pl_fit:
        y = fit.power_law.ccdf()
        perc = len(fit.data)/len(fit.data_original)
#         fit.power_law.plot_ccdf(linestyle='--', color='purple', ax=ax)
        plt.plot(fit.data, y * perc, linestyle='--', color='purple')
        plt.axvline(fit.xmin, linestyle='--', color='r', label=f'xmin: {fit.xmin}')
#         plt.axvline(fit.xmax, linestyle='--', color='pink', label=f'xmax: {fit.xmax}')


    if vlines > 0:  # plot like quartile lines for number of nodes.
        # rough q-tiles
        q = vlines
        colors = plt.cm.rainbow(np.linspace(0, 1, q))
        rev_dd = list(reversed(dd))
        for i in range(1, q):
            plt.axvline(rev_dd[i * len(dd)//q], label=f'qtile-{i}/{q}', c=colors[i])
        plt.legend()

if __name__ == '__main__':
    import os
    names = ['avg_deg_locality', 'avg_detour_dist', 'avg_dist', 'closure_weak', 'closure', 'clustering_coeff', 'degeneracy', 'size']
    # reverse sort names
    names = sorted(names, reverse=True)
    for name in names:
        print(name)
        print(pd.read_csv(p + 'output_data/' + name + '.csv').shape)

    dfs = []
    for name in names:
        dfs.append(pd.read_csv(p + 'output_data/' + name + '.csv'))

    df_stats = pd.concat(dfs, axis=1)
    # write to csv
    df_stats.to_csv(p + 'output_data/overall_df_stats.csv', index=False)

    # check if every column of name graph is identical
    rows = list(df_stats.graph.iterrows())
    rows_unique = [set(x[1].unique()) for x in rows]
    print(set([len(x) for x in rows_unique]))

    # remove all but the first column of name 'graph' in df_stats
    number_of_graphs_columns = df_stats.graph.shape[1]
    # non graph columns
    df_stats_non_graph = df_stats.loc[:, df_stats.columns != 'graph']
    df_stats= pd.concat([df_stats.graph.iloc[:, 0], df_stats_non_graph], axis=1)

    # write to csv
    df_stats.to_csv(p + 'output_data/overall_df_stats.csv', index=False)


    print(f'input_names_all: {len(input_names_all)}')
    print(f'input_names_girg_deg_scaling: {len(input_names_girg_deg_scaling)}')
    print(f'input_names_gen_high_deg: {len(input_names_gen_high_deg)}')
    print(f'input_names: {len(input_names)}')
    print(f'input_names_girg: {len(input_names_girg)}')
    print(f'input_names_cl: {len(input_names_cl)}')
    print(f'input_names_er: {len(input_names_er)}')
    print(f'input_names_real: {len(input_names_real)}')