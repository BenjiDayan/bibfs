import gc
import random

import random

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
        self.queue = {0: ListDict([s])}
        self.seen = {s: 0}
        self.expanded = set()
        self.node_to_parent = {s: None}


class BiBFS:
    def __init__(self, g, s, t):
        self.sBFS = BFS(g, s)
        self.tBFS = BFS(g, t)
        self.g = g

    def terminate(self, bfs, other_bfs, intersection_point):
        # return other_bfs.seen[w] + 1 + layer
        bfs_path = []
        other_bfs_path = []

        v = bfs.node_to_parent[intersection_point]
        while v is not None:
            bfs_path.append(v)
            v = bfs.node_to_parent[v]

        v = other_bfs.node_to_parent[intersection_point]
        while v is not None:
            other_bfs_path.append(v)
            v = other_bfs.node_to_parent[v]

        self.path = bfs_path[::-1] + other_bfs_path
        self.intersection_point = intersection_point

    def run(self):
        if self.sBFS.s == self.tBFS.s:
            self.path = [self.sBFS.s]
            self.intersection_point = self.sBFS.s
            return True

        while self.sBFS.queue and self.tBFS.queue:
            # TODO change to checking lengths of queues
            s_queue_size = sum([len(self.sBFS.queue[layer]) for layer in self.sBFS.queue])
            t_queue_size = sum([len(self.tBFS.queue[layer]) for layer in self.tBFS.queue])
            if s_queue_size <= t_queue_size:
                bfs = self.sBFS
                other_bfs = self.tBFS
            else:
                bfs = self.tBFS
                other_bfs = self.sBFS
            # pop a random element from the list of the current layer
            layer = list(bfs.queue.keys())[0]

            v = bfs.queue[layer].choose_random_item()
            bfs.queue[layer].remove(v)
            bfs.expanded.add(v)

            # delete i layer if exhausted; i+1 remains
            if len(bfs.queue[layer]) == 0:
                del bfs.queue[layer]

            for w in self.g.iterNeighbors(v):
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

        return False

class BiBFS_ExactExpandSmallerQueue(BiBFS):
    def terminate(self, bfs_potential, other_bfs_potential, intersection_point_potential):
        try:
            layer1 = list(self.sBFS.queue.keys())[0]
            layer2 = list(self.tBFS.queue.keys())[0]
        except IndexError:  # one of the queues is empty. direct edge guaranteed
            super().terminate(bfs_potential, other_bfs_potential, intersection_point_potential)
            return

        # pick the smaller queue front
        if len(self.sBFS.queue[layer1]) > len(self.tBFS.queue[layer2]):
            bfs, other_bfs = self.sBFS, self.tBFS
        else:
            bfs, other_bfs = self.tBFS, self.sBFS

        # expand all nodes in the smaller queue front, check for edge to other queue front
        layer1 = list(bfs.queue.keys())[0]
        layer2 = list(other_bfs.queue.keys())[0]
        for v in bfs.queue[layer1]:
            for w in self.g.iterNeighbors(v):
                if w in other_bfs.queue[layer2]:
                    bfs.node_to_parent[w] = v
                    super().terminate(bfs, other_bfs, w)
                    return

        super().terminate(bfs_potential, other_bfs_potential, intersection_point_potential)


class BiBFS_ExactCheckDirectEdges(BiBFS):
    def terminate(self, bfs_potential, other_bfs_potential, intersection_point_potential):
        try:
            layer1 = list(self.sBFS.queue.keys())[0]
            layer2 = list(self.tBFS.queue.keys())[0]
        except IndexError:  # one of the queues is empty. direct edge guaranteed
            super().terminate(bfs_potential, other_bfs_potential, intersection_point_potential)
            return

        # check for direct edges
        for u in self.sBFS.queue[layer1]:
            for v in self.tBFS.queue[layer2]:
                if self.g.hasEdge(u, v):
                    self.sBFS.node_to_parent[v] = u
                    super().terminate(self.sBFS, self.tBFS, v)
                    return

        super().terminate(bfs_potential, other_bfs_potential, intersection_point_potential)
        return

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

