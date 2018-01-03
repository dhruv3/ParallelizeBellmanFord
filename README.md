# Parallelize Bellman Ford

Three algorithms have been implemented and they can run in different configurations. Parameters and values required to run them have been explained in next paragraph. Additionally, you can refer [Report 1](https://github.com/dhruv3/ParallelizeBellmanFord/blob/master/sssp_report.pdf) and [Report 2](https://github.com/dhruv3/ParallelizeBellmanFord/blob/master/sssp_opt_report.pdf) to see the detailed analysis done on the three algorithms.
To run the code do the following steps:

1. Compile using the command 'make sssp'

2. Run the program by giving the command-

	./sssp --input ama.txt --bsize 1024 --bcount 2 --output output.txt --method opt --usesmem no --sync outcore

Variables:
ama.txt is the name of valid graph file. It is expected to be the same folder.

bsize is the block size

bcount represents the block number

output.txt is the txt file generated that contains the final answer

opt represents the optimized method. You can also set it to bmf and tpe to run bellman ford method and work efficient method respectively.

usesmem represents if shared memory should be used or not. Shared memory method only implemented for tpe and bmf algorithms

sync value could be outcore or incore. Outcore uses an auxiliary data structure whereas incore only use single data structure.
