

import benji_utils as utils
import bbbfs_algorithms as bbbfa

import networkit as nk
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

from scipy.stats.mstats import winsorize
import statsmodels.api as sm
lowess = sm.nonparametric.lowess

import re

import matplotlib.pyplot as plt
import numpy as np

def load_data(p = '../M3_ext_val_data/girg_scaling_queue_fixed/', regex=True):
    # p = '../M3_ext_val_data/chunglu_scaling/'
    # p = '../M3_ext_val_data/chunglu_scaling_LBF_VBSQB/'
    # p = '../M3_ext_val_data/girg_scaling_queue_fixed/'
    import glob
    dfs = []

    algo_name_map = {
        'python-BiBFS_LayerbalancedFull': 'LBF',
        'python-BiBFS_ExactExpandSmallerQueueBetter': 'VBSQB',
        'python-BiBFS_Layerbalanced': 'LB',
        'python-BiBFS_VertexBalancedApproximate': 'VBA',
        'python-BiBFS_ExactExpandSmallerQueue': 'VBSQ',
        'python-BiBFS_ExactCheckDirectEdges': 'VBDE',
        'python-BiBFS_EdgeBalancedApproximate': 'EBA',
    }


    for fn in glob.glob(p + '*.csv'):
        df = (pd.read_csv(fn))
        df.algo = df.algo.apply(lambda x: algo_name_map[x])
        dfs.append(df)

    df = pd.concat(dfs)

    infos = {}
    for fn in glob.glob(p + '*_info.txt'):
        # name = fn[:fn.index('_info.txt')]
        name = re.match('.*?(\w*_n.*)_info.txt', fn.split('/')[-1]).group(1)
        with open(fn) as f:
            # print(f.read())
            infos[name] = f.read().split()
            infos[name] = (int(infos[name][0]), int(infos[name][2]))

    df_infos = pd.DataFrame(infos).T
    df_infos
    df_infos.columns = ['n', 'm']
    # make index a column called graph
    df_infos['graph'] = df_infos.index
    df_infos.reset_index(inplace=True, drop=True)
    df_infos

    df = df.merge(df_infos, on='graph', how='left')

    if regex:
        df['n_orig'] = df.graph.apply(lambda x: int(re.search('n=(.*?)_', x).group(1)))
        df['ple'] = df.graph.apply(lambda x: re.search('ple=(.*?)_', x).group(1))
        df['ple'] = df['ple'].astype(float)
        df['seed'] = df.graph.apply(lambda x: re.search('seed=(.*?)$', x).group(1))
        df['deg'] = df.graph.apply(lambda x: int(re.search('deg=(.*?)_', x).group(1)))
        if 'alpha' in df.graph.iloc[0]:
            df['alpha'] = df.graph.apply(lambda x: float(re.search('alpha=(.*?)_', x).group(1)))
        if 'dim' in df.graph.iloc[0]:
            df['dim'] = df.graph.apply(lambda x: int(re.search('dim=(.*?)_', x).group(1)))

    return df



def get_algo_medians(df, keys=['n_orig', 'deg', 'ple', 'alpha']):
    # search space median for each algo
    graph_algo_medians = df.groupby(keys + ['algo']).search_space.median()
    graph_algo_medians = graph_algo_medians.reset_index()

    # median and std of m for each graph
    graph_m_medians = df.groupby(keys).m.median()
    graph_m_stds = df.groupby(keys).m.std()

    graph_algo_medians = graph_algo_medians.merge(graph_m_medians, on=keys, how='left')
    graph_algo_medians = graph_algo_medians.merge(graph_m_stds, on=keys, how='left')
    graph_algo_medians.rename(columns={'m_x': 'm', 'm_y': 'm_std'}, inplace=True)

    # median and std of n for each graph
    graph_n_medians = df.groupby(keys).n.median()
    graph_n_stds = df.groupby(keys).n.std()
    graph_algo_medians = graph_algo_medians.merge(graph_n_medians, on=keys, how='left')
    graph_algo_medians = graph_algo_medians.merge(graph_n_stds, on=keys, how='left')
    graph_algo_medians.rename(columns={'n_x': 'n', 'n_y': 'n_std'}, inplace=True)

    # calculate m_goal
    graph_algo_medians['m_goal'] = graph_algo_medians.deg.astype(int) * graph_algo_medians.n_orig.astype(int) / 2

    # std of search space for each algo
    graph_algo_medians = graph_algo_medians.merge(df.groupby(keys + ['algo']).search_space.std(), on=keys + ['algo'], how='left')


def get_algo_search_space_medians(df, keys=['n_orig', 'deg', 'ple', 'alpha'], max_degrees=False):
    graph_grped = df.groupby(keys)

    a = pd.concat([graph_grped.m.median(), graph_grped.m.std(), graph_grped.n.median(), graph_grped.n.std()], axis=1)#
    a.columns = ['m', 'm_std', 'n', 'n_std']


    graph_grped2 = df.groupby(keys + ['algo'])
    graph_algo_search_space_df = pd.concat([
        graph_grped2.search_space.median(),
        graph_grped2.search_space.std()], axis=1)

    graph_algo_search_space_df.columns = ['search_space', 'search_space_std']


    if max_degrees:
        b = pd.concat([graph_grped2.med_frac.median(), graph_grped2.fd_frac.median(), graph_grped2.med_md_ratio.median()], axis=1)
        b.columns = ['med_frac', 'fd_frac', 'med_md_ratio']
        graph_algo_search_space_df = graph_algo_search_space_df.merge(b, on=keys + ['algo'], how='left')

    out = graph_algo_search_space_df
    # make indices columns
    out.reset_index(inplace=True)
    out = out.merge(a, on=keys, how='left')


    out['m_goal'] = out.deg.astype(int) * out.n_orig.astype(int) / 2
    out['search_space_fraction'] = out['search_space'] / out['m']
    out.sort_values(keys + ['algo'], inplace=True)
    return out


def plot_df(df, degrees=[10, 20, 30], alpha=0.7):
    """
    # E.g. for a CL subset:
    df = foo.loc[(foo.n_orig == n) & (foo.ple <= 3)]
    """
    
    # 3 rows, 1 column
    fig, axes = plt.subplots(len(degrees), 1, figsize=(6, 6*len(degrees)))

    for i, deg in enumerate(degrees):
        ax = axes[i] if isinstance(axes, list) else axes  # Get the appropriate subplot axis
        df_sub = df.loc[df.deg == deg]

        # VBA algorithm
        c = df_sub.loc[df_sub.algo == 'VBA'].groupby('ple').search_space.median()
        m = df_sub.loc[df_sub.algo == 'VBA'].groupby('ple').m.median()
        ax.plot(c.index, np.log(c.values) / np.log(m.values), label='VBA', marker='o', alpha=alpha)

        # VBSQB algorithm
        c = df_sub.loc[df_sub.algo == 'VBSQB'].groupby('ple').search_space.median()
        m = df_sub.loc[df_sub.algo == 'VBSQB'].groupby('ple').m.median()
        ax.plot(c.index, np.log(c.values) / np.log(m.values), label='VBE', marker='o', alpha=alpha)

        # LBF algorithm
        c = df_sub.loc[df_sub.algo == 'LBF'].groupby('ple').search_space.median()
        m = df_sub.loc[df_sub.algo == 'LBF'].groupby('ple').m.median()
        ax.plot(c.index, np.log(c.values) / np.log(m.values), label='LB', marker='o', alpha=alpha)

        # LB algorithm
        c = df_sub.loc[df_sub.algo == 'LB'].groupby('ple').search_space.median()
        m = df_sub.loc[df_sub.algo == 'LB'].groupby('ple').m.median()
        ax.plot(c.index, np.log(c.values) / np.log(m.values), label='LBES', marker='o', alpha=alpha)


        # Theoretical bounds
        tau = np.linspace(2.0, 3.0, 100)
        # (tau-2)/(tau-1) curve
        # y = (tau - 2) / (tau - 1)
        ax.plot(tau, (tau - 2) / (tau - 1), label='VBA bound', ls='--', alpha=alpha)
        ax.plot(tau, tau-tau + 1/2, label='VBE bound', ls='--', alpha=alpha)
        ax.plot(tau, (4-tau)/2, label='LB bound', ls='--', alpha=alpha)


        ax.set_xlabel('power-law exponent')
        ax.set_ylabel('runtime exponent')
        ax.set_ylim(0.0, 0.8)
        # ax.set_title(f'Median c = m^x exponent x for n={n} deg={deg} ple=tau GIRG graphs')
        ax.legend()



    plt.tight_layout()
    return fig, axes


def fixed_ple_n_scaling_plots(df_sub, ple=2.0, fig=None, algo='VBA', bounds=True, shift=1.0, color=None, alpha=0.7):
    """
    df_sub = df.loc[(df.algo == algo) & (df.ple == ple) & (df.deg == deg)]
    """
    if not fig:
        fig = plt.figure(figsize=(6, 6))
    ax = plt.gca()

    y = df_sub.search_space
    x = df_sub.m
    if color:
        ax.scatter(x, y, label=algo, marker='o', color=color, alpha=alpha)
    else:
        ax.scatter(x, y, label=algo, marker='o', alpha=alpha)
    ax.set_xlabel('# edges')
    ax.set_ylabel('cost')
    # ax.set_title(f'ple={ple}, deg={deg}, algo={algo}')
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Insert line of gradient ple that passes through the first point
    if bounds:
        x_mid = x.median()
        i_mid = np.argwhere(x == x_mid).flatten()[0]
        y_mid = y.iloc[i_mid]
        y2 = x**((ple-2)/(ple-1))
        y2 *= y_mid * shift/ y2.iloc[i_mid]
        ax.plot(x, y2, label='VBA bound', color='#9467bd', ls='--', alpha=alpha)

        y3 = x**(0.5)
        y3 *= y_mid *shift / y3.iloc[i_mid]
        ax.plot(x, y3, label='VBE bound', color='#8c564b', ls='--', alpha=alpha)

    ax.legend()

    return fig, ax


    if __name__ == '__main__':
        girg_df = load_data(p = '../M3_ext_val_data/girg_scaling_queue_fixed/')
        graph_algo_search_space_medians = get_algo_search_space_medians(girg_df, keys=['n_orig', 'deg', 'ple', 'alpha'])
        
        gassms = graph_algo_search_space_medians
        df = gassms.loc[(gassms.n_orig == 80000) & (gassms.alpha == 2.3) & (gassms.ple <=4)]
        fig, axes = plot_df(df, degrees=[10, 20, 30], n=80000)

        # Save figure
        fig.savefig('girg_scaling.png')

        chunglu_df = load_data(p = '../M3_ext_val_data/chunglu_scaling_queue_fixed/')
        graph_algo_search_space_medians = get_algo_search_space_medians(chunglu_df, keys=['n_orig', 'deg', 'ple'])
        gassms = graph_algo_search_space_medians
        df = gassms.loc[(gassms.n_orig == 80000) & (gassms.ple <=4)]
        fig, axes = plot_df(df, degrees=[10, 30, 60], n=80000)
        fig.savefig('chunglu_scaling.png')

