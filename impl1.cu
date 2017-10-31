#include <vector>
#include <iostream>

#include "utils.h"
#include "cuda_error_check.cuh"
#include "initial_graph.hpp"
#include "parse_graph.hpp"

__global__ void pulling_kernel(std::vector<initial_vertex> * graph, int offset, int * anyChange){

    //update me based on my neighbors. Toggle anyChange as needed.
    //offset will tell you who I am.
}

void puller(std::vector<initial_vertex> * graph, int blockSize, int blockNum){
    unsigned int *initDist, *distance_cur, *distance_prev; 
	int *anyChange;
	int *hostAnyChange = (int*)malloc(sizeof(int));
	graph_node *edge_list, *L;
	unsigned int edge_num;
	
	edge_num = count_edges(*graph);
	edge_list = (graph_node*) malloc(sizeof(graph_node)*edge_num);
	initDist = (unsigned int*)calloc(graph->size(),sizeof(unsigned int));	
	pull_distances(initDist, graph->size());
	pull_edges(*graph, edge_list, edge_num);

	//sort by source vertex
	qsort(edge_list, edge_num, sizeof(graph_node), cmp_edge);			

	unsigned int *hostDistanceCur = new unsigned int[graph->size()];

	cudaMalloc((void**)&distance_cur, (size_t)sizeof(unsigned int)*(graph->size()));
	cudaMalloc((void**)&distance_prev, (size_t)sizeof(unsigned int)*(graph->size()));
	cudaMalloc((void**)&anyChange, (size_t)sizeof(int));
	cudaMalloc((void**)&L, (size_t)sizeof(graph_node)*edge_num);

	cudaMemcpy(distance_cur, initDist, (size_t)sizeof(unsigned int)*(graph->size()), cudaMemcpyHostToDevice);
	cudaMemcpy(distance_prev, initDist, (size_t)sizeof(unsigned int)*(graph->size()), cudaMemcpyHostToDevice);
	cudaMemcpy(L, edge_list, (size_t)sizeof(graph_node)*edge_num, cudaMemcpyHostToDevice);
	
	cudaMemset(anyChange, 0, (size_t)sizeof(int));

    setTime();

    /*
     * Do all the things here!
     **/

    std::cout << "Took " << getTime() << "ms.\n";
}
