#include <vector>
#include <iostream>

#include "utils.h"
#include "cuda_error_check.cuh"
#include "initial_graph.hpp"
#include "parse_graph.hpp"

//get total edges
unsigned int total_edges(std::vector<initial_vertex>& graph){
	unsigned int edge_counter = 0;
	for(int i = 0 ; i < graph.size() ; i++){
	    edge_counter += graph[i].nbrs.size();
	}
	return edge_counter;
}

int cmp_edge(const void *a, const void *b){	
	return ( (int)(((graph_node *)a)->srcIndex) - (int)(((graph_node *)b)->srcIndex));
}

void set_edges(std::vector<initial_vertex>& graph, edge_node* edge_list, unsigned int edge_counter){
	unsigned int k = 0;
	for(int i = 0 ; i < graph.size() ; i++){
	    for(int j = 0 ; j < graph[i].nbrs.size() ; j++, k++){
			edge_list[k].srcIndex = graph[i].nbrs[j].srcIndex;
			edge_list[k].destIndex = i;
			edge_list[k].weight = graph[i].nbrs[j].edgeValue.weight;
	    }
	}
}

void set_distances(unsigned int* initDist, int size){
	initDist[0] = 0;
	for(int i = 1; i < size; i++){
	    initDist[i] = UINT_MAX; 
	}
}

__global__ void pulling_kernel(std::vector<initial_vertex> * graph, int offset, int * anyChange){

    //update me based on my neighbors. Toggle anyChange as needed.
    //offset will tell you who I am.
}

__global__ void edge_process(const edge_node *L, const unsigned int edge_num, unsigned int *distance_prev, unsigned int *distance_cur, int *anyChange){
	
	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	int total_threads = blockDim.x * gridDim.x;
	int warp_id = thread_id/32;
	if(total_threads % 32 == 0){
		int warp_num = total_threads/32;
	}
	else{
		int warp_num = total_threads/32 + 1;
	}
	int lane_id = thread_id % 32;
	
	//given in the psuedocode
	int load = (edge_num % warp_num == 0) ? edge_num/warp_num : edge_num/warp_num+1;
	int beg = load * warp_id;
	int end = min(edge_num, beg + load);
	beg = beg + lane_id;

	unsigned int u, v, w;
	for(int i = beg; i < end; i+=32){
		u = L[i].src;
		v = L[i].dst;
		w = L[i].weight;
		if(distance_prev[u] == UINT_MAX){
			continue;
		} 
		else if(distance_prev[u] + w < distance_cur[v]){
			anyChange[0] = 1;
			atomicMin(&distance_cur[v], distance_prev[u] + w);
		}
	}
}

void puller(std::vector<initial_vertex> * graph, int blockSize, int blockNum){
    unsigned int *initDist, *distance_cur, *distance_prev; 
	int *anyChange;
	int *hostAnyChange = (int*)malloc(sizeof(int));
	
	graph_node *edge_list, *L;
	unsigned int edge_counter;
	edge_counter = total_edges(*graph);
	edge_list = (graph_node*) malloc(sizeof(graph_node)*edge_counter);
	
	//TODO: calloc changed to malloc
	initDist = (unsigned int*)malloc(sizeof(unsigned int)*graph->size());	
	set_distances(initDist, graph->size());
	set_edges(*graph, edge_list, edge_counter);

	//sort by source vertex
	//http://www.cplusplus.com/reference/cstdlib/qsort/
	qsort(edge_list, edge_counter, sizeof(graph_node), cmp_edge);			

	unsigned int *hostDistanceCur = new unsigned int[graph->size()];

	cudaMalloc((void**)&distance_cur, (size_t)sizeof(unsigned int)*(graph->size()));
	cudaMalloc((void**)&distance_prev, (size_t)sizeof(unsigned int)*(graph->size()));
	cudaMalloc((void**)&anyChange, (size_t)sizeof(int));
	cudaMalloc((void**)&L, (size_t)sizeof(graph_node)*edge_counter);

	cudaMemcpy(distance_cur, initDist, (size_t)sizeof(unsigned int)*(graph->size()), cudaMemcpyHostToDevice);
	cudaMemcpy(distance_prev, initDist, (size_t)sizeof(unsigned int)*(graph->size()), cudaMemcpyHostToDevice);
	cudaMemcpy(L, edge_list, (size_t)sizeof(graph_node)*edge_counter, cudaMemcpyHostToDevice);
	
	cudaMemset(anyChange, 0, (size_t)sizeof(int));

    setTime();

	for(int i=0; i < ((int) graph->size())-1; i++){
		edge_process<<<blockNum,blockSize>>>(L, edge_num, distance_prev, distance_cur, anyChange);
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

	cudaFree(distance_cur);
	cudaFree(distance_prev);
	cudaFree(anyChange);
	cudaFree(L);
	
	delete[] hostDistanceCur;
	free(initDist);
	free(edge_list);
}
