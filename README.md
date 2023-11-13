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



## 11/11/2023

### Trying to compile our bfs code

```Consolidate compiler generated dependencies of target deg_distr
[ 54%] Built target deg_distr
Consolidate compiler generated dependencies of target dist
[ 55%] Building CXX object CMakeFiles/dist.dir/cli/dist.cpp.o
[ 56%] Linking CXX executable dist
/usr/lib/gcc/x86_64-alpine-linux-musl/11.2.1/../../../../x86_64-alpine-linux-musl/bin/ld: CMakeFiles/dist.dir/cli/dist.cpp.o: warning: relocation against `_ZTV9BFSBiNode' in read-only section `.text._ZNSt17_Function_handlerIFSt4pairIjjEjjEZ9dist_algoI9BFSBiNodeEDaRK5GraphEUljjE_E10_M_managerERSt9_Any_dataRKSA_St18_Manager_operation[_ZNSt17_Function_handlerIFSt4pairIjjEjjEZ9dist_algoI9BFSBiNodeEDaRK5GraphEUljjE_E10_M_managerERSt9_Any_dataRKSA_St18_Manager_operation]'
/usr/lib/gcc/x86_64-alpine-linux-musl/11.2.1/../../../../x86_64-alpine-linux-musl/bin/ld: CMakeFiles/dist.dir/cli/dist.cpp.o: in function `dist(std::filesystem::__cxx11::path, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, unsigned int)':
dist.cpp:(.text+0x1566): undefined reference to `vtable for BFSBiNodeBalanced'
/usr/lib/gcc/x86_64-alpine-linux-musl/11.2.1/../../../../x86_64-alpine-linux-musl/bin/ld: dist.cpp:(.text+0x181b): undefined reference to `vtable for BFSBiNode'
/usr/lib/gcc/x86_64-alpine-linux-musl/11.2.1/../../../../x86_64-alpine-linux-musl/bin/ld: CMakeFiles/dist.dir/cli/dist.cpp.o: in function `std::_Function_handler<std::pair<unsigned int, unsigned int> (unsigned int, unsigned int), dist_algo<BFSBiNodeBalanced>(Graph const&)::{lambda(unsigned int, unsigned int)#1}>::_M_manager(std::_Any_data&, std::_Any_data const&, std::_Manager_operation)':
dist.cpp:(.text._ZNSt17_Function_handlerIFSt4pairIjjEjjEZ9dist_algoI17BFSBiNodeBalancedEDaRK5GraphEUljjE_E10_M_managerERSt9_Any_dataRKSA_St18_Manager_operation[_ZNSt17_Function_handlerIFSt4pairIjjEjjEZ9dist_algoI17BFSBiNodeBalancedEDaRK5GraphEUljjE_E10_M_managerERSt9_Any_dataRKSA_St18_Manager_operation]+0xa6): undefined reference to `vtable for BFSBiNodeBalanced'
/usr/lib/gcc/x86_64-alpine-linux-musl/11.2.1/../../../../x86_64-alpine-linux-musl/bin/ld: CMakeFiles/dist.dir/cli/dist.cpp.o: in function `std::_Function_handler<std::pair<unsigned int, unsigned int> (unsigned int, unsigned int), dist_algo<BFSBiNode>(Graph const&)::{lambda(unsigned int, unsigned int)#1}>::_M_manager(std::_Any_data&, std::_Any_data const&, std::_Manager_operation)':
dist.cpp:(.text._ZNSt17_Function_handlerIFSt4pairIjjEjjEZ9dist_algoI9BFSBiNodeEDaRK5GraphEUljjE_E10_M_managerERSt9_Any_dataRKSA_St18_Manager_operation[_ZNSt17_Function_handlerIFSt4pairIjjEjjEZ9dist_algoI9BFSBiNodeEDaRK5GraphEUljjE_E10_M_managerERSt9_Any_dataRKSA_St18_Manager_operation]+0xa6): undefined reference to `vtable for BFSBiNode'
/usr/lib/gcc/x86_64-alpine-linux-musl/11.2.1/../../../../x86_64-alpine-linux-musl/bin/ld: warning: creating DT_TEXTREL in a PIE
collect2: error: ld returned 1 exit status
make[2]: *** [CMakeFiles/dist.dir/build.make:98: dist] Error 1
make[1]: *** [CMakeFiles/Makefile2:524: CMakeFiles/dist.dir/all] Error 2
make: *** [Makefile:146: all] Error 2
/ext_val
```

For now we will get rid of bfs_bi_node_balanced (.hpp, .cpp) as it's distracting and I don't know the original intention anyway

`code/release/dist --algo bfs_bi_balanced --pairs 4 --seed 123 --no-header input_data/adj_array/ex10`
code/release/dist --algo bfs_bi_node --pairs 4 --seed 123 --no-header input_data/adj_array/ex10

Ok so now what do the numbers mean?

dist(input file, algo, nr_pairs): 
- randomly picks a nr_pairs (i, j) in (n-1)
- outputs: bfs_bi_node,123,1214,772,14,0.061316,22124
- algo, seed, s, t, res.first, timer, res.second
- res = compute_dist(s,t)
- compute_dist = dist_algo<BFSBiBalanced>(G);
    -     unsigned dist = algo(G, s, t);
          return std::make_pair(dist, algo.search_space());
    - m_search_space is the total number of nodes that have been added to a queue


- before fixing the bfs_bi_node_exact_bug
```
bash-5.1# code/release/dist --algo bfs_bi_node --pairs 4 --seed 123 --no-header input_data/adj_array/ex10
bfs_bi_node,123,1,276,5,0.007504,558
bfs_bi_node,123,1450,639,15,0.076525,25178
bfs_bi_node,123,818,1438,12,0.054504,23062
bfs_bi_node,123,1214,772,14,0.051939,22124
bash-5.1# code/release/dist --algo bfs_bi_node_exact --pairs 4 --seed 123 --no-header input_data/adj_array/ex10
bfs_bi_node_exact,123,1,276,5,0.008456,575
bfs_bi_node_exact,123,1450,639,15,0.079992,25205
bfs_bi_node_exact,123,818,1438,12,0.055415,23089
bfs_bi_node_exact,123,1214,772,14,0.059182,22213
```