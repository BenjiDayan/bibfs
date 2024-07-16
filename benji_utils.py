import pandas as pd
import os

df = pd.read_csv('../dist.csv')


p = '../M3_ext_val_data/'
graph_names = os.listdir(p + 'edge_list')
graph_names.sort()
import networkit as nk

# g = nk.Graph(p + 'edge_list/DD_g63')
import glob
edgelists = glob.glob(p + 'edge_list/*')
edgelists[:10]

def graph_name_to_nk(graph_name):
    fn = edgelists[edgelists.index(f'{p}edge_list/{graph_name}')]
    return nk.graphio.EdgeListReader(' ', 0).read(fn)

def slice_df(df, s, t, name):
    return df.loc[(df.s==s) & (df.t == t) & (df.graph == name)]


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

