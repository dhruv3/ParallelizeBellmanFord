#include <vector>
#include <iostream>

#include "utils.h"
#include "cuda_error_check.cuh"
#include "initial_graph.hpp"
#include "parse_graph.hpp"

__global__ void neighborHandling_kernel(std::vector<initial_vertex> * peeps, int offset, int * anyChange){

    //update me based on my neighbors. Toggle anyChange as needed.
    //Enqueue and dequeue me as needed.
    //Offset will tell you who I am.
}

void puller_incore_impl2(std::vector<initial_vertex> * graph, int blockSize, int blockNum, ofstream &outputFile){
	unsigned int *initDist, *distance_cur, *distance_prev; 
	int *anyChange;
	//todo
	int *hostAnyChange = (int*)malloc(sizeof(int));
	graph_node *edge_list, *L;
	unsigned int edge_counter;
	edge_counter = total_edges(*graph);
	edge_list = (graph_node*) malloc(sizeof(graph_node)*edge_counter);
	
	//TODO: calloc changed to malloc
	initDist = (unsigned int*)malloc(sizeof(unsigned int)*graph->size());	
	initDist[0] = 0;
	for(int i = 1; i < graph->size(); i++){
	    initDist[i] = UINT_MAX; 
	}
	//set_edges(*graph, edge_list, edge_counter);
	unsigned int k = 0;
	for(int i = 0 ; i < graph->size() ; i++){
		std::vector<neighbor> nbrs = (*graph)[i].nbrs;
	    for(int j = 0 ; j < nbrs.size() ; j++, k++){
			edge_list[k].src = nbrs[j].srcIndex;
			edge_list[k].dst = i;
			edge_list[k].weight = nbrs[j].edgeValue.weight;
	    }
	}

	//sort by source vertex
	//http://www.cplusplus.com/reference/cstdlib/qsort/
	qsort(edge_list, edge_counter, sizeof(graph_node), cmp_edge);			

	//todo
	unsigned int *hostDistanceCur = new unsigned int[graph->size()];

	cudaMalloc((void**)&L, (size_t)sizeof(graph_node)*edge_counter);
	cudaMalloc((void**)&distance_cur, (size_t)sizeof(unsigned int)*(graph->size()));
	cudaMalloc((void**)&distance_prev, (size_t)sizeof(unsigned int)*(graph->size()));
	cudaMalloc((void**)&anyChange, (size_t)sizeof(int));
	

	cudaMemcpy(distance_cur, initDist, (size_t)sizeof(unsigned int)*(graph->size()), cudaMemcpyHostToDevice);
	cudaMemcpy(distance_prev, initDist, (size_t)sizeof(unsigned int)*(graph->size()), cudaMemcpyHostToDevice);
	cudaMemcpy(L, edge_list, (size_t)sizeof(graph_node)*edge_counter, cudaMemcpyHostToDevice);
	
	cudaMemset(anyChange, 0, (size_t)sizeof(int));

    setTime();

	for(int i=0; i < ((int) graph->size())-1; i++){
		edge_process<<<blockNum,blockSize>>>(L, edge_counter, distance_prev, distance_cur, anyChange);
		cudaMemcpy(hostAnyChange, anyChange, sizeof(int), cudaMemcpyDeviceToHost);
		if(!hostAnyChange[0]){
			break;
		} 
		else {
			cudaMemset(anyChange, 0, (size_t)sizeof(int));
			cudaMemcpy(hostDistanceCur, distance_cur, (sizeof(unsigned int))*(graph->size()), cudaMemcpyDeviceToHost);
			cudaMemcpy(distance_cur, distance_prev, (sizeof(unsigned int))*(graph->size()), cudaMemcpyDeviceToDevice);
			cudaMemcpy(distance_prev, hostDistanceCur,(sizeof(unsigned int))*(graph->size()), cudaMemcpyHostToDevice);
		}
	}

	cout << "Took " << getTime() << "ms.\n";

	cudaMemcpy(hostDistanceCur, distance_cur, (sizeof(unsigned int))*(graph->size()), cudaMemcpyDeviceToHost);

	for(int i=0; i < graph->size(); i++){
		if(hostDistanceCur[i] == UINT_MAX){
		    outputFile << i << ":" << "INF" << endl;
		}
		else{
		    outputFile << i << ":" << hostDistanceCur[i] << endl; 
		}
	}

	cudaFree(L);
	cudaFree(distance_cur);
	cudaFree(distance_prev);
	cudaFree(anyChange);
	
	delete[] hostDistanceCur;
	free(initDist);
	free(edge_list);
}