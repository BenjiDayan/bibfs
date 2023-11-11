# bibfs

## 09/11/2023
I think our work is in docker container named ext_val3. See its output_data folder

- Note it seems that for some experiments we need to be root to write files??

```bash
PermissionError: [Errno 13] Permission denied: 'output_data/avg_dist_tests'

```

So I think `docker exec -u root -it ext_val3 /bin/bash` is the way to go. This did manage to create a bunch of root folders in my home directory that I can't delete now :(

error we got when doing `./experiments.py algo`

```
WARNING: unexpected return code (105) for command: code/release/dist --algo bfs_bi_node_balanced --pairs 100 --seed 3404785993 --no-header input_data/adj_array/hospital-ward-proximity
        --algo:  not in {bfs,bfs_bi_balanced,bfs_bi_always_swap}
        Run with --help for more information.
```

But I thinkt he first three did work fine.