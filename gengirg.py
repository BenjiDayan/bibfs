import subprocess
import tqdm
from concurrent.future import ProcessPoolExecutor

n = 80000
dims = [1, 2, 3]
ples = [2.01, 2.1, 2.3,  2.5, 2.7, 2.9, 3.0]
seeds = [1,2,3]
degs = [10, 20, 30]
alphas = [1, 1.5, 2.3, 5, float('inf')]

from itertools import product
params = list(product(ples, seeds, degs, dims, alphas))

def run(args):
    ple, seed, deg, dim, alpha = args
    foo = f"girgs2/girg_n={n}_deg={deg}_dim={dim}_ple={ple}_alpha={alpha}_seed={seed}"
    print(foo)
    subprocess.run(["gengirg", "-n", str(n), "-deg", str(deg), "-d", str(dim), "-ple", str(ple), "-alpha", str(alpha),
                    "-wseed", str(seed), "-pseed", str(seed), "-sseed", str(seed), "-edge", "1", "-file", foo])


# for ple, seed, deg, dim, alpha in tqdm.tqdm(params):
#     foo = f"girgs/girg_n={n}_deg={deg}_dim={dim}_ple={ple}_alpha={alpha}_seed={seed}"
#     print(foo)
#     # subprocess.run(["gengirg", f"-n {n}", f"-deg {deg}", f"-d {dim}", f"-ple {ple}", f"-alpha {alpha}",
#     #                 f"-wseed {seed}", f"-pseed {seed}", f"-sseed {seed}", "-edge 1", f"-file '{foo}'"])
#     # print(f"python3 gengirg.py --n={n} --deg={deg} --ple={ple} --seed={seed} --output={foo}")
#     # break

#     subprocess.run(["gengirg", "-n", str(n), "-deg", str(deg), "-d", str(dim), "-ple", str(ple), "-alpha", str(alpha),
#                     "-wseed", str(seed), "-pseed", str(seed), "-sseed", str(seed), "-edge", "1", "-file", foo])

if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=6) as executor:
        list(tqdm.tqdm(executor.map(lambda p: run(*p), params), total=len(params)))