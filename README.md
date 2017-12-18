# ParallelizeBellmanFord

To run the code do the following steps:

1. Compile using the command 'make sssp'

2. Run the program by giving the command-

	./sssp --input ama.txt --bsize 1024 --bcount 2 --output output.txt --method opt --usesmem no --sync outcore

Variables:
ama.txt is the name of valid graph file. It is expected to be the same folder.
bsize is the block size
bcount represents the block number
output.txt is the txt file generated that contains the final answer
opt represents the optimized method.
usesmem should be set to no.
sync should be set to outcore.
