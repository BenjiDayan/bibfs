import gc
import os
import random

import random
import time
import networkit as nk
import networkx as nx
import pandas as pd
from dataclasses import dataclass
import numpy as np
import benji_utils as utils
import generators
import concurrent.futures
from tqdm import tqdm

from networkit.graph import Graph

# I don't know why we use this... is it just a data structure that we can choose
# a random item from efficiently? It's literally a set that we can sample a random
# elt from
# It's actually kinda like a list dict hybrid huh.
class ListDict(object):
    def __init__(self, intial_items=[]):
        self.item_to_position = {}
        self.items = []
        for item in intial_items:
            self.add(item)

    def getfirst(self):
        return self.items[0]

    def __contains__(self, item):
        return item in self.item_to_position

    def __iter__(self):
        return iter(self.items)

    def add(self, item):
        if item in self.item_to_position:
            return
        self.items.append(item)
        self.item_to_position[item] = len(self.items)-1

    def remove(self, item):
        """Remove an item from the list. O(1) time."""
        position = self.item_to_position.pop(item)
        last_item = self.items.pop()
        if position != len(self.items):
            self.items[position] = last_item
            self.item_to_position[last_item] = position

    def choose_random_item(self):
        return random.choice(self.items)
    
    def __len__(self):
        return len(self.items)
    
    def __repr__(self):
        return str(self.items)

from collections import OrderedDict
class Queue:
    def __init__(self, initial_data=None):
        self.data = OrderedDict()
        if initial_data is not None:
            for item in initial_data:
                self.add(item)
    def add(self, item):
        """Actually enqueue an item"""
        key = item
        if key in self.data:
            self.data.move_to_end(key)
        self.data[key] = True
    def dequeue(self):
        try:
            return self.data.popitem(last=False)[0]
        except KeyError:
            print("Empty queue")
    def remove(self, item):
        try:
            del self.data[item]
        except KeyError:
            print(f"Item {item} not in queue")
    def __len__(self):
        return len(self.data)
    def __repr__(self):
        return f"Queue({self.data.items()})"
    def __iter__(self):
        return iter(self.data.keys())
    def __contains__(self, item):
        return item in self.data


def approx_average_case_old(g, s, t):
    if s == t:
        return 0, 1, 1
    
    queue_s = {0 : [s]}
    queue_t = {0 : [t]}
    seen_s = {s: 0}
    seen_t = {t: 0}
    while queue_s and queue_t: # if either queue is empty, we're done
        queue_chosen = None
        other_queue = None
        seen_chosen = None
        other_seen = None
        if len(seen_s) <= len(seen_t):
            queue_chosen = queue_s
            seen_chosen = seen_s
            other_queue = queue_t
            other_seen = seen_t
        else:
            queue_chosen = queue_t
            seen_chosen = seen_t
            other_queue = queue_s
            other_seen = seen_s
        # pop a random element from the list of the current layer
        layer = list(queue_chosen.keys())[0]
        v = queue_chosen[layer].pop(random.randint(0, len(queue_chosen[layer])-1))
        
        # delete i layer if exhausted; i+1 remains
        if len(queue_chosen[layer]) == 0:
            del queue_chosen[layer]  

        for w in g.iterNeighbors(v):
            if w in other_seen:
                return other_seen[w] + 1 + layer, len(seen_s), len(seen_t)
            if w not in seen_chosen:
                seen_chosen[w] = layer+1
                if layer+1 not in queue_chosen:
                    queue_chosen[layer+1] = []
                queue_chosen[layer+1].append(w)
    return -1, len(seen_s), len(seen_t)

class BFS:
    def __init__(self, g, s):
        self.s = s
        # In general queue[layer_i] gets eventually converted into
        # seen[w in layer_i] = i.
        # queue will only have up to two active keys at a time, layer_i and layer_i+1
        self.queue = {0: Queue([s])}
        self.seen = {s: 0}
        self.expanded = set()
        self.node_to_parent = {s: None}
        self.work = 0  # we'll up this count whenever we explore a new edge.


# I think this is equivalent to the vertex-balanced BFS for approximate shortest path
class BiBFS:
    def __init__(self, g, s, t):
        self.sBFS = BFS(g, s)
        self.tBFS = BFS(g, t)
        self.g = g

    def terminate(self, bfs, other_bfs, intersection_point=None):
        self.work = self.sBFS.work + self.tBFS.work
        if intersection_point is None:
            return

        # return other_bfs.seen[w] + 1 + layer
        bfs_path = []
        other_bfs_path = []

        # bfs_path is [s_k, ..., s_1, s_0 = s]
        v = bfs.node_to_parent[intersection_point]
        while v is not None:
            bfs_path.append(v)
            v = bfs.node_to_parent[v]

        # other_bfs_path is [t_l, t_l-1, ..., t_1, t_0 = t]
        v = other_bfs.node_to_parent[intersection_point]
        while v is not None:
            other_bfs_path.append(v)
            v = other_bfs.node_to_parent[v]

        # i.e. path = s_0=s, ..., s_k, intersection_point, t_l, ..., t_0=t
        self.path = bfs_path[::-1] + [intersection_point] + other_bfs_path
        self.intersection_point = intersection_point
        self.dist = len(self.path) - 1

    def run(self):
        if self.sBFS.s == self.tBFS.s:
            self.terminate(self.sBFS, self.tBFS, self.sBFS.s)
            return True

        return self.run_expansion_search()


        

class BiBFS_VertexBalancedApproximate(BiBFS):

    def choose_bfs_smaller_queue(self):
        # We choose the one with a smaller queue.
        s_queue_size = sum([len(self.sBFS.queue[layer]) for layer in self.sBFS.queue])
        t_queue_size = sum([len(self.tBFS.queue[layer]) for layer in self.tBFS.queue])

        if s_queue_size <= t_queue_size:
            return self.sBFS, self.tBFS
        else:
            return self.tBFS, self.sBFS
        

    def choose_bfs(self):
        # Choose by which has the smaller number seen
        num_seen_s = len(self.sBFS.seen)
        num_seen_t = len(self.tBFS.seen)

        if num_seen_s <= num_seen_t:
            return self.sBFS, self.tBFS
        else:
            return self.tBFS, self.sBFS
        
    def run_expansion_search(self):
        while self.sBFS.queue and self.tBFS.queue:
            bfs, other_bfs = self.choose_bfs()
            # pop a random element from the list of the current layer
            layer = list(bfs.queue.keys())[0]

            # v = bfs.queue[layer].choose_random_item()
            # bfs.queue[layer].remove(v)
            v = bfs.queue[layer].dequeue()
            bfs.expanded.add(v)


            for w in self.g.iterNeighbors(v):
                bfs.work += 1
                if w in other_bfs.seen:  # either a hop or a direct edge - terminate?
                    bfs.seen[w] = layer + 1
                    bfs.node_to_parent[w] = v
                    self.terminate(bfs, other_bfs, w)
                    return True

                if w not in bfs.seen:
                    bfs.seen[w] = layer + 1
                    bfs.node_to_parent[w] = v
                    if layer + 1 not in bfs.queue:
                        bfs.queue[layer + 1] = Queue([])
                    bfs.queue[layer + 1].add(w)

            # delete i layer if exhausted; i+1 remains
            if len(bfs.queue[layer]) == 0:
                del bfs.queue[layer]

        self.terminate(bfs, other_bfs, intersection_point=None)
        return False

class BiBFS_ExactExpandSmallerQueue(BiBFS_VertexBalancedApproximate):
    def terminate(self, bfs_potential, other_bfs_potential, intersection_point_potential):
        try:
            layer1 = list(self.sBFS.queue.keys())[0]
            layer2 = list(self.tBFS.queue.keys())[0]
        except IndexError:  # one of the queues is empty. direct edge guaranteed
            super().terminate(bfs_potential, other_bfs_potential, intersection_point_potential)
            return

        # pick the smaller queue front - layer1 is the smaller one
        if len(self.sBFS.queue[layer1]) < len(self.tBFS.queue[layer2]):
            bfs, other_bfs = self.sBFS, self.tBFS
        else:
            bfs, other_bfs = self.tBFS, self.sBFS
            layer1, layer2 = layer2, layer1

        # expand all nodes in the smaller queue front, check for edge to other queue front
        for v in bfs.queue[layer1]:
            for w in self.g.iterNeighbors(v):
                bfs.work += 1
                if w in other_bfs.queue[layer2]:
                    bfs.node_to_parent[w] = v
                    super().terminate(bfs, other_bfs, w)
                    return

        super().terminate(bfs_potential, other_bfs_potential, intersection_point_potential)


class BiBFS_ExactExpandSmallerQueueBetter(BiBFS_ExactExpandSmallerQueue):#
    """We actually don't need to expand smaller queue front if the intersection point was from
    inner to inner. We can just terminate immediately."""
    def terminate(self, bfs_potential, other_bfs_potential, intersection_point_potential):
        try:
            layer1 = list(self.sBFS.queue.keys())[0]
            layer2 = list(self.tBFS.queue.keys())[0]
        except IndexError:  # one of the queues is empty. direct edge guaranteed
            super().terminate(bfs_potential, other_bfs_potential, intersection_point_potential)
            return
        
        s_dist = self.sBFS.seen[intersection_point_potential]
        t_dist = self.tBFS.seen[intersection_point_potential]
        if s_dist == layer1 or t_dist == layer2:
            # super super terminate
            return BiBFS_VertexBalancedApproximate.terminate(self, self.sBFS, self.tBFS, intersection_point_potential)
    
        return super().terminate(bfs_potential, other_bfs_potential, intersection_point_potential)


class BiBFS_ExactCheckDirectEdges(BiBFS_VertexBalancedApproximate):
    def terminate(self, bfs_potential, other_bfs_potential, intersection_point_potential):
        try:
            layer1 = list(self.sBFS.queue.keys())[0]
            layer2 = list(self.tBFS.queue.keys())[0]
        except IndexError:  # one of the queues is empty. direct edge guaranteed
            super().terminate(bfs_potential, other_bfs_potential, intersection_point_potential)
            return

        # check for direct edges
        extra_work = 0
        for u in self.sBFS.queue[layer1]:
            for v in self.tBFS.queue[layer2]:
                extra_work += 1
                if self.g.hasEdge(u, v):
                    self.sBFS.node_to_parent[v] = u
                    super().terminate(self.sBFS, self.tBFS, v)
                    self.work += extra_work
                    return

        super().terminate(bfs_potential, other_bfs_potential, intersection_point_potential)
        self.work += extra_work
        return
    
class BiBFS_EdgeBalancedApproximate(BiBFS):

    @dataclass
    class Container:
        bfs: BFS
        v_curr: int
        E_curr: Queue

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.container_s = {'bfs': self.sBFS, 'v_curr': self.sBFS.s, 'E_curr': self.g.edges(self.s)}
        # self.container_t = {'bfs': self.tBFS, 'v_curr': self.tBFS.s, 'E_curr': self.g.edges(self.t)}

        self.container_s = self.Container(self.sBFS, None, Queue([]))
        self.container_t = self.Container(self.tBFS, None, Queue([]))

        self.edge_visited_count = 0
        # self.s_curr = self.s
        # self.t_curr = self.t
        # self.Es_curr = self.g.edges(self.s)
        # self.Et_curr = self.g.edges(self.t)

    def get_containers(self):
        if self.edge_visited_count % 2 == 0:
            return self.container_s, self.container_t
        else:
            return self.container_t, self.container_s

        
    # TODO finish this - it's not complete.
    def run_expansion_search(self):
        while (self.sBFS.queue and self.tBFS.queue):
            c, c_other = self.get_containers()
            bfs, other_bfs = c.bfs, c_other.bfs
            layer = list(bfs.queue.keys())[0]

            # if the current edge list is empty, get a new one
            if len(c.E_curr) == 0:
                # c.v_curr = bfs.queue[layer].choose_random_item()
                # bfs.queue[layer].remove(c.v_curr)
                c.v_curr = bfs.queue[layer].dequeue()
                bfs.expanded.add(c.v_curr)
                c.E_curr = Queue(self.g.iterNeighbors(c.v_curr))
                

            # pop a random edge from the current edge list
            # w = c.E_curr.choose_random_item()
            # c.E_curr.remove(w)
            w = c.E_curr.dequeue()
            self.edge_visited_count += 1
            bfs.work += 1
            if w in other_bfs.seen:
                bfs.node_to_parent[w] = c.v_curr
                self.terminate(bfs, other_bfs, w)
                return True
            if w not in bfs.seen:
                bfs.seen[w] = layer + 1
                bfs.node_to_parent[w] = c.v_curr
                if layer + 1 not in bfs.queue:
                    bfs.queue[layer + 1] = Queue([])
                bfs.queue[layer + 1].add(w)
            
            # if the queue layer i is exhausted and so is the current edge list, delete layer i
            # to move onto layer i+1
            if len(bfs.queue[layer]) == 0 and len(c.E_curr) == 0:
                    del bfs.queue[layer]

        self.terminate(bfs, other_bfs, intersection_point=None)
        return False


class BiBFS_Layerbalanced(BiBFS_VertexBalancedApproximate):
    def choose_bfs(self):
        # We choose the one with a smaller queue.
        # There should only be one layer however.
        s_queue_size = sum([len(self.sBFS.queue[layer]) for layer in self.sBFS.queue])
        t_queue_size = sum([len(self.tBFS.queue[layer]) for layer in self.tBFS.queue])

        if s_queue_size <= t_queue_size:
            return self.sBFS, self.tBFS
        else:
            return self.tBFS, self.sBFS
        
    def run_expansion_search(self):
        layer = None
        while self.sBFS.queue and self.tBFS.queue:
            # just started BiBFS, or just deleted layer.
            if layer is None or layer not in bfs.queue:
                bfs, other_bfs = self.choose_bfs()
                layer = list(bfs.queue.keys())[0]

            # pop a random element from the list of the current layer
            # v = bfs.queue[layer].choose_random_item()
            # bfs.queue[layer].remove(v)
            v = bfs.queue[layer].dequeue()
            bfs.expanded.add(v)


            for w in self.g.iterNeighbors(v):
                bfs.work += 1  #
                if w in other_bfs.seen:  # either a hop or a direct edge - terminate?
                    bfs.node_to_parent[w] = v
                    self.terminate(bfs, other_bfs, w)
                    return True

                if w not in bfs.seen:
                    bfs.seen[w] = layer + 1
                    bfs.node_to_parent[w] = v
                    if layer + 1 not in bfs.queue:
                        bfs.queue[layer + 1] = Queue([])
                    bfs.queue[layer + 1].add(w)

            # delete i layer if exhausted; i+1 remains
            if len(bfs.queue[layer]) == 0:
                del bfs.queue[layer]

        self.terminate(bfs, other_bfs, intersection_point=None)
        return False
    
class BiBFS_LayerbalancedFull(BiBFS_Layerbalanced):
    """Like LB, but will finish expanding the current layer even after finding an intersection point"""
    def run_expansion_search(self):
        layer = None
        found_intersection_point = None
        while self.sBFS.queue and self.tBFS.queue:
            # just started BiBFS, or just deleted layer.
            if layer is None or layer not in bfs.queue:
                if found_intersection_point is not None:
                    self.terminate(bfs, other_bfs, found_intersection_point)
                    return True
                bfs, other_bfs = self.choose_bfs()
                layer = list(bfs.queue.keys())[0]

            # pop a random element from the list of the current layer
            # v = bfs.queue[layer].choose_random_item()
            # bfs.queue[layer].remove(v)
            v = bfs.queue[layer].dequeue()
            bfs.expanded.add(v)


            for w in self.g.iterNeighbors(v):
                bfs.work += 1  #
                if w in other_bfs.seen:  # either a hop or a direct edge - terminate?
                    bfs.node_to_parent[w] = v
                    # self.terminate(bfs, other_bfs, w)
                    # return True
                    found_intersection_point = w

                if w not in bfs.seen:
                    bfs.seen[w] = layer + 1
                    bfs.node_to_parent[w] = v
                    if layer + 1 not in bfs.queue:
                        bfs.queue[layer + 1] = Queue([])
                    bfs.queue[layer + 1].add(w)

            # delete i layer if exhausted; i+1 remains
            if len(bfs.queue[layer]) == 0:
                del bfs.queue[layer]

        if found_intersection_point is not None:
            self.terminate(bfs, other_bfs, found_intersection_point)
            return True

        self.terminate(bfs, other_bfs, intersection_point=None)
        return False

def approx_average_case(g, s, t):
    if s == t:
        return 0, 1, 1, 0
    
    max_deg_expanded = 0

    layer_degrees_s = {s: g.degree(s)}
    layer_degrees_t = {t: g.degree(t)}
    queue_s = {0 : ListDict([s])}
    queue_t = {0 : ListDict([t])}
    seen_s = {s: 0}
    seen_t = {t: 0}
    while queue_s and queue_t: # if either queue is empty, we're done
        queue_chosen = None
        other_queue = None
        seen_chosen = None
        other_seen = None
        # TODO change to checking lengths of queues
        s_queue_size = sum([len(queue_s[layer]) for layer in queue_s])
        t_queue_size = sum([len(queue_t[layer]) for layer in queue_t])
        if s_queue_size <= t_queue_size:
            queue_chosen = queue_s
            seen_chosen = seen_s
            other_seen = seen_t
            layer_degrees_chosen = layer_degrees_s
        else:
            queue_chosen = queue_t
            seen_chosen = seen_t
            other_seen = seen_s
            layer_degrees_chosen = layer_degrees_t
        # pop a random element from the list of the current layer
        layer = list(queue_chosen.keys())[0]

        # print(queue_chosen)
        # v = queue_chosen[layer].pop(random.randint(0, len(queue_chosen[layer])-1))
        # v = queue_chosen[layer].pop()
        v = queue_chosen[layer].choose_random_item()
        queue_chosen[layer].remove(v)
        
        # delete i layer if exhausted; i+1 remains
        if len(queue_chosen[layer]) == 0:
            del queue_chosen[layer]  

        for w in g.iterNeighbors(v):
            if w in other_seen:
                return other_seen[w] + 1 + layer, len(seen_s), len(seen_t), max_deg_expanded
            if w not in seen_chosen:
                seen_chosen[w] = layer+1
                layer_degrees_chosen[w] = g.degree(w)
                if layer+1 not in queue_chosen:
                    queue_chosen[layer+1] = ListDict([])
                queue_chosen[layer+1].add(w)

        # print(f'after: {queue_chosen}')

        max_deg_expanded = max(max_deg_expanded, g.degree(v))

    return -1, len(seen_s), len(seen_t), max_deg_expanded


import benji_utils as utils

def run_for_row(row, algo=BiBFS):
    s, t = row.s, row.t
    seed = row.seed
    random.seed(seed)
    nk.setSeed(seed, True)
    g = utils.graph_name_to_nk(row.graph)

    algo = algo(g, s, t)
    st = time.time()
    found = algo.run()
    rt = time.time() - st

    new_row = row.copy()
    new_row['algo'] =  f'python-{algo.__class__.__name__}'
    new_row['search_space'] = algo.work
    new_row['dist'] = algo.dist if found else -1
    new_row['time_dist'] = rt

    print(f"{row.graph} {s}-{t}; dist: {new_row['dist']} search_space: {new_row['search_space']}")
    return new_row

def run_for_rows(rows, algos):
    new_rows = []
    g = None
    graph_name = None

    for row in rows:
        s, t = row.s, row.t
        seed = row.seed
        random.seed(seed)
        nk.setSeed(seed, True)
        if row.graph != graph_name:
            graph_name = row.graph
            g = utils.graph_name_to_nk(row.graph)
        
        for a in algos:
            algo = a(g, s, t)
            st = time.time()
            found = algo.run()
            rt = time.time() - st

            new_row = row.copy()
            new_row['algo'] =  f'python-{algo.__class__.__name__}'
            new_row['search_space'] = algo.work
            new_row['dist'] = algo.dist if found else -1
            new_row['time_dist'] = rt

            new_rows.append(new_row)
    return new_rows


def replicate_experiment(num_graphs=5):
    df = utils.df
    graph_names = df.graph.unique()
    new_rows = []
    for gn in graph_names[:num_graphs]:
        # print(gn)
        for i, row in df[(df.graph == gn) & (df.algo == 'bfs_bi_node')].iterrows():
            for algo in [BiBFS_VertexBalancedApproximate, BiBFS_ExactExpandSmallerQueue, BiBFS_ExactCheckDirectEdges, BiBFS_EdgeBalancedApproximate]:
                new_row = run_for_row(row, algo)
                new_rows.append(new_row)

    return pd.DataFrame(new_rows)



def random_sample_pair(n):
    return tuple(sorted(random.sample(range(n), 2)))

def random_sample_pairs(n, num_pairs):
    if num_pairs > n*(n-1)/4:
        return np.random.permutation(all_pairs(n))[:num_pairs]
    else:
        out = set()
        while len(out) < num_pairs:
            out.add(random_sample_pair(n))
        return list(out)

def all_pairs(n):
    return [(i, j) for i in range(n) for j in range(i+1, n)]


def run_for_pairs(g, pairs, algo_class=BiBFS_VertexBalancedApproximate, do_print=False):
    new_rows = []
    for s, t in pairs if not do_print else tqdm(pairs):
        algo = algo_class(g, s, t)
        st = time.time()
        found = algo.run()
        rt = time.time() - st

        new_row = {'s': s, 't': t, 'algo': f'python-{algo.__class__.__name__}', 'search_space': algo.work, 'dist': algo.dist if found else -1, 'time_dist': rt}
        new_rows.append(new_row)
    return new_rows


# Does a comparison of algos on a graph vs CL counterpart
def do_comparison(g, seed=42, fit_iters=3, n_pairs=1000, g_cl=None,
                  algos=[BiBFS_Layerbalanced, BiBFS_VertexBalancedApproximate, BiBFS_ExactExpandSmallerQueue, BiBFS_ExactCheckDirectEdges, BiBFS_EdgeBalancedApproximate],
                  do_print=False):
    random.seed(seed)
    nk.setSeed(seed, True)

    tau = 2.5
    if not g_cl:
        g_cl = generators.fit_connected_chunglu_to_g(g, tau=tau, iters=fit_iters)
    
    n, e, n2, e2 = g.numberOfNodes(), g.numberOfEdges(), g_cl.numberOfNodes(), g_cl.numberOfEdges()
    print(f'nodes: {n} {n2} edges: {e} {e2}')
    if np.abs((n-n2)/n) > 0.1 or np.abs((e-e2)/e) > 0.1:
        print('Greater than 10% difference!')
        return None

    # pick n_pairs random pairs of nodes without replacement
    random.seed(seed)
    nk.setSeed(seed, True)
    n = min(g.numberOfNodes(), g_cl.numberOfNodes())
    pairs = random_sample_pairs(n, n_pairs)
    rows = []
    for algo in algos:
        rows += run_for_pairs(g, pairs, algo, do_print=do_print)
        rows += run_for_pairs(g_cl, pairs, algo, do_print=do_print)
    df = pd.DataFrame(rows)
    # rows = run_for_pairs(g, pairs, BiBFS_VertexBalancedApproximate)
    # rows_cl = run_for_pairs(g_cl, pairs, BiBFS_VertexBalancedApproximate)
    # df = pd.DataFrame(rows + rows_cl)
    return df#

def run_algos_on_g(g, seed=42, n_pairs=1000,
        algos=[BiBFS_Layerbalanced, BiBFS_LayerbalancedFull, BiBFS_VertexBalancedApproximate, BiBFS_ExactExpandSmallerQueueBetter, BiBFS_ExactExpandSmallerQueue, BiBFS_ExactCheckDirectEdges, BiBFS_EdgeBalancedApproximate],
        do_print=False):
    random.seed(seed)
    nk.setSeed(seed, True)
    n, e = g.numberOfNodes(), g.numberOfEdges()
    print(f'nodes: {n} edges: {e}')
    pairs = random_sample_pairs(n, n_pairs)
    rows = []
    for algo in algos:
        rows += run_for_pairs(g, pairs, algo, do_print=do_print)
    df = pd.DataFrame(rows)
    return df

def do_comparison2(g_name):
    g = utils.graph_name_to_nk(g_name)
    df = do_comparison(g)
    df['graph'] = g_name
    return df

# do comparison with pre-loaded CL graph.
def do_real_fake_comparison(g_name, **kwargs):
    g = utils.graph_name_to_nk(g_name)
    try:
        g_cl = utils.graph_name_to_nk(g_name, cl=True)
    except:
        print(f'No cl graph for {g_name}')
        return
    df = do_comparison(g, g_cl=g_cl, **kwargs)
    df['graph'] = g_name
    return df




def generate_cl_counterpart(name, tau=None, seed=42):

    g = utils.graph_name_to_nk(name)
    print(name, g.numberOfNodes(), g.numberOfEdges())
    np.random.seed(seed=seed)
    random.seed(seed)
    nk.setSeed(seed, True)
    try:
        g_cl, fit = generators.fit_connected_chunglu_to_g(g, tau=tau, iters=3, tol=0.1)
    except Exception as e:
        # this seems to happen if powerlaw couldn't fit an alpha, it instead returns
        # alpha=nan, and then when we generate weights those are nans too :(
        # BZR-MD is a tiny graph of 33 nodes and 528 edges this occurs
        print(f"Errored on {name}")
        return
    if not fit:
        print(f"Could not fit {name}")
        return

    path = utils.p + 'edge_list_cl_taufit/'
    os.makedirs(path, exist_ok=True)
    out_fp = f'{path}{name}_cl'
    print(out_fp)
    nk.graphio.EdgeListWriter(' ', 0).write(g_cl, out_fp)


def generate_cl_counterparts():
    graph_names = sorted(utils.input_names_real)
    graph_names = graph_names

    # for name in graph_names:
    #     print(name)
    #     g = utils.graph_name_to_nk(name)
    #     print(name, g.numberOfNodes(), g.numberOfEdges())
    #     random.seed(seed)
    #     nk.setSeed(seed, True)
    #     g_cl, fit = generators.fit_connected_chunglu_to_g(g, tau=tau, iters=3, tol=0.1)
    #     if not fit:
    #         print(f"Could not fit {name}")
    #         continue

    #     out_fp = f'{utils.p}edge_list_cl/{name}_cl'
    #     print(out_fp)
    #     nk.graphio.EdgeListWriter(' ', 0).write(g_cl, out_fp)

    # Actually process pool executor this
    with concurrent.futures.ProcessPoolExecutor(max_workers=14) as executor:
        # Wrap the executor.map call with tqdm
        results = list(tqdm(executor.map(generate_cl_counterpart, graph_names), total=len(graph_names)))



if __name__ == '__main__':
    df = utils.df
    # iloc 500 sized chunks of the dataframe
    df_graphs = []
    for i in range(0, len(df), 500):
        df_graphs.append(df.iloc[i:i+500])

    # make sure each graph is in its own dataframe
    seen = set()
    for df_graph in df_graphs:
        assert len(df_graph.graph.unique()) == 1
        assert df_graph.graph.unique()[0] not in seen
        seen.add(df_graph.graph.unique()[0])


    # Assuming run_for_rows and df_graphs are defined
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        # Wrap the executor.map call with tqdm
        results = list(tqdm(executor.map(run_for_rows, [[x for _, x in df_graph.iterrows()] for df_graph in df_graphs[:20]], 
                                        [[BiBFS_VertexBalancedApproximate, 
                                        BiBFS_ExactExpandSmallerQueue, 
                                        BiBFS_ExactCheckDirectEdges, 
                                        BiBFS_EdgeBalancedApproximate]]*20), total=len(df_graphs[:20])))

