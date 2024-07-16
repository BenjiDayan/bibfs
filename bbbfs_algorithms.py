import gc
import random

import random
import time
import networkit as nk
import pandas as pd
from dataclasses import dataclass

# I don't know why we use this... is it just a data structure that we can choose
# a random item from efficiently? It's literally a set that we can sample a random
# elt from
class ListDict(object):
    def __init__(self, intial_items=[]):
        self.item_to_position = {}
        self.items = []
        for item in intial_items:
            self.add(item)

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
        self.queue = {0: ListDict([s])}
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

            v = bfs.queue[layer].choose_random_item()
            bfs.queue[layer].remove(v)
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
                        bfs.queue[layer + 1] = ListDict([])
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
        if len(self.sBFS.queue[layer1]) > len(self.tBFS.queue[layer2]):
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
        E_curr: ListDict

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.container_s = {'bfs': self.sBFS, 'v_curr': self.sBFS.s, 'E_curr': self.g.edges(self.s)}
        # self.container_t = {'bfs': self.tBFS, 'v_curr': self.tBFS.s, 'E_curr': self.g.edges(self.t)}

        self.container_s = self.Container(self.sBFS, None, ListDict([]))
        self.container_t = self.Container(self.tBFS, None, ListDict([]))

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
                c.v_curr = bfs.queue[layer].choose_random_item()
                bfs.queue[layer].remove(c.v_curr)
                bfs.expanded.add(c.v_curr)
                c.E_curr = ListDict(self.g.iterNeighbors(c.v_curr))
                

            # pop a random edge from the current edge list
            w = c.E_curr.choose_random_item()
            c.E_curr.remove(w)
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
                    bfs.queue[layer + 1] = ListDict([])
                bfs.queue[layer + 1].add(w)
            
            # if the queue layer i is exhausted and so is the current edge list, delete layer i
            # to move onto layer i+1
            if len(bfs.queue[layer]) == 0 and len(c.E_curr) == 0:
                    del bfs.queue[layer]

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
