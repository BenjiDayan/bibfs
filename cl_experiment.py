import networkit as nk
import numpy as np
import bbbfs_algorithms
import matplotlib.pyplot as plt

def powerlaw_dist(tau=2.5, x_min=1, n=1000):
    """sample from a tau exponent power law distribution
    pdf: prop to x^-(a+1), i.e. tau = a+1
    mean: ((tau-1) x_min)/(tau - 2) for tau > 2
    x_min: support is [x_min, inf]
    size: number of samples to draw
    """
    a = tau-1
    pareto = (np.random.pareto(a, size=n) + 1) * x_min
    return pareto

# @profile
def do_stuff(tau, ns):
    avg_work_list = []
    avg_max_degree_expanded_list = []
    # for n in [1000, 10000, 100000, 200000, 400000, 700000]:
    for n in ns:
        print(n)
        for _ in range(1):
            print(_)
            degree_seq = 2 * powerlaw_dist(tau=tau, x_min=1, n=n)

            gen = nk.generators.ChungLuGenerator(degree_seq)

            g = gen.generate()

        # nk.overview(g)


            pairs = 1000
            us = np.random.choice(n, size=pairs)
            vs = np.random.choice(n, size=pairs)
            # dists = np.zeros((len(us), len(us)))
            avg_works = []
            max_degs_expanded = []

            for i, u in enumerate(us):
                v = vs[i]
                    # if j <= i:
                    #     continue
                    # thing =  nk.distance.Dijkstra(g, u, True, False, v).run()
                    # dists[i, j] = thing.distance(v)
                print('--------')
                dist, ls, lt, max_degree_expanded =  bbbfs_algorithms.approx_average_case(g, u, v)
                avg_works.append(ls + lt)
                max_degs_expanded.append(max_degree_expanded)

        avg_work = np.mean(avg_works)
        avg_max_degree_expanded = np.mean(max_degs_expanded)
        # avg_max_degree_expanded = (np.sum(max_degs_expanded))/((k*(k-1)/2))

        print(f'avg_work: {avg_work}')
        print(f'log_n(avg_work): {np.emath.logn(n, avg_work)}')
        avg_work_list.append((n, avg_work))

        avg_max_degree_expanded_list.append(avg_max_degree_expanded)



    ns = np.array([x[0] for x in avg_work_list])
    plt.plot(ns, [x[1] for x in avg_work_list], label='avg work', marker='o')
    plt.plot(ns, ns**((tau-2)/(tau-1)), label='n^((tau-2)/(tau-1))')
    plt.plot(ns, ns**(1/2), label='n^(1/2)')
    plt.plot(ns, avg_max_degree_expanded_list, label='max degree expanded', marker='o')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()

# do_stuff(2.3, list(map(int, [1000, 10000, 100000, 200000, 400000, 700000, 1e6, 3e6])))
# do_stuff(2.3, [1000])

if __name__ == '__main__':
