#include <vector>
#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <algorithm>

#include "utils.h"
#include "cuda_error_check.cuh"
#include "initial_graph.hpp"
#include "parse_graph.hpp"

using namespace std;


//get total edges
unsigned int total_edges(std::vector<initial_vertex>& graph){
	unsigned int edge_counter = 0;
	for(int i = 0 ; i < graph.size() ; i++){
	    edge_counter += graph[i].nbrs.size();
	}
	return edge_counter;
}

//comparator function used by qsort
int cmp_edge(const void *a, const void *b){	
	return ( (((graph_node *)a)->src) - (((graph_node *)b)->src));
}

int cmp_edge_dst(const void *a, const void *b){	
	return ( (((graph_node *)a)->dst) - (((graph_node *)b)->dst));
}

//outcore
//kernel outcore method
__global__ void edge_process(const graph_node *L, const unsigned int edge_counter, unsigned int *distance_prev, unsigned int *distance_cur, int *anyChange){
	
	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	int total_threads = blockDim.x * gridDim.x;
	int warp_id = thread_id/32;
	int warp_num;
	if(total_threads % 32 == 0){
		warp_num = total_threads/32;
	}
	else{
		warp_num = total_threads/32 + 1;
	}
	int lane_id = thread_id % 32;
	
	//given in the psuedocode
	int load = (edge_counter % warp_num == 0) ? edge_counter/warp_num : edge_counter/warp_num+1;
	int beg = load * warp_id;
	int end = beg + load;
	if(edge_counter < beg + load)
		end = edge_counter;
	beg = beg + lane_id;

	unsigned int u, v, w;
	for(int i = beg; i < end; i+=32){
		u = L[i].src;
		v = L[i].dst;
		w = L[i].weight;
		if(distance_prev[u] != UINT_MAX && distance_prev[u] + w < distance_cur[v]){
			atomicMin(&distance_cur[v], distance_prev[u] + w);
			anyChange[0] = 1;
		}
	}
}

//outcore device method
void puller(std::vector<initial_vertex> * graph, int blockSize, int blockNum, ofstream &outputFile){
	unsigned int *distance_cur, *distance_prev; 
    
    unsigned int *initDist;
	//set initial distance to max except for source node
	initDist = (unsigned int*)malloc(sizeof(unsigned int)*graph->size());	
	initDist[0] = 0;
	for(int i = 1; i < graph->size(); i++){
	    initDist[i] = UINT_MAX; 
	}
	
	unsigned int edge_counter;
	edge_counter = total_edges(*graph);
	graph_node *edge_list;
	edge_list = (graph_node*) malloc(sizeof(graph_node)*edge_counter);
	//for each member of edge list set initial values
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
	//sort by dst: qsort(edge_list, edge_counter, sizeof(graph_node), cmp_edge_dst);
	
	graph_node *L;			
	unsigned int *swapDistVariable = new unsigned int[graph->size()];
	int *anyChange, check_if_change;

	cudaMalloc((void**)&L, (size_t)sizeof(graph_node)*edge_counter);
	cudaMalloc((void**)&distance_cur, (size_t)sizeof(unsigned int)*(graph->size()));
	cudaMalloc((void**)&distance_prev, (size_t)sizeof(unsigned int)*(graph->size()));
	cudaMalloc((void**)&anyChange, (size_t)sizeof(int));

	cudaMemcpy(distance_cur, initDist, (size_t)sizeof(unsigned int)*(graph->size()), cudaMemcpyHostToDevice);
	cudaMemcpy(distance_prev, initDist, (size_t)sizeof(unsigned int)*(graph->size()), cudaMemcpyHostToDevice);
	cudaMemcpy(L, edge_list, (size_t)sizeof(graph_node)*edge_counter, cudaMemcpyHostToDevice);
	cudaMemset(anyChange, 0, (size_t)sizeof(int));

    setTime();

	for(int i=0; i < (graph->size())-1; i++){
		edge_process<<<blockNum,blockSize>>>(L, edge_counter, distance_prev, distance_cur, anyChange);
		cudaMemcpy(&check_if_change, anyChange, sizeof(int), cudaMemcpyDeviceToHost);
		if(check_if_change == 0){
			break;
		} 
		cudaMemset(anyChange, 0, (size_t)sizeof(int));
		cudaMemcpy(swapDistVariable, distance_cur, (sizeof(unsigned int))*(graph->size()), cudaMemcpyDeviceToHost);
		cudaMemcpy(distance_cur, distance_prev, (sizeof(unsigned int))*(graph->size()), cudaMemcpyDeviceToDevice);
		cudaMemcpy(distance_prev, swapDistVariable,(sizeof(unsigned int))*(graph->size()), cudaMemcpyHostToDevice);
	}

	cout << "Took " << getTime() << "ms.\n";

	cudaMemcpy(swapDistVariable, distance_cur, (sizeof(unsigned int))*(graph->size()), cudaMemcpyDeviceToHost);

	for(int i=0; i < graph->size(); i++){
		if(swapDistVariable[i] == UINT_MAX){
		    outputFile << i << ":" << "INF" << endl;
		}
		else{
		    outputFile << i << ":" << swapDistVariable[i] << endl; 
		}
	}

	cudaFree(L);
	cudaFree(distance_cur);
	cudaFree(distance_prev);
	cudaFree(anyChange);
	
	delete[] swapDistVariable;
	free(initDist);
	free(edge_list);
}

//incore
//incore kernel method
__global__ void edge_process_incore(const graph_node *L, const unsigned int edge_counter, unsigned int *distance, int *anyChange){
	
	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	int total_threads = blockDim.x * gridDim.x;
	int warp_id = thread_id/32;
	int warp_num;
	if(total_threads % 32 == 0){
		warp_num = total_threads/32;
	}
	else{
		warp_num = total_threads/32 + 1;
	}
	int lane_id = thread_id % 32;
	
	int load = (edge_counter % warp_num == 0) ? edge_counter/warp_num : edge_counter/warp_num+1;
	int beg = load * warp_id;
	int end = beg + load;
	if(edge_counter < beg + load)
		end = edge_counter;
	beg = beg + lane_id;

	unsigned int u, v, w;
	for(int i = beg; i < end; i+=32){
		u = L[i].src;
		v = L[i].dst;
		w = L[i].weight;
		if(distance[u] != UINT_MAX && distance[u] + w < distance[v]){
			atomicMin(&distance[v], distance[u] + w);
			anyChange[0] = 1;
		}
	}
}

//incore device method
void puller_incore(vector<initial_vertex> * graph, int blockSize, int blockNum, ofstream &outputFile){

	unsigned int *distance; 
	unsigned int *initDist;
	//set initial distance to max except for source node
	initDist = (unsigned int*)malloc(sizeof(unsigned int)*graph->size());	
	initDist[0] = 0;
	for(int i = 1; i < graph->size(); i++){
	    initDist[i] = UINT_MAX; 
	}

	unsigned int edge_counter;
	edge_counter = total_edges(*graph);
	graph_node *edge_list;
	edge_list = (graph_node*) malloc(sizeof(graph_node)*edge_counter);
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
	//sort by dst: qsort(edge_list, edge_counter, sizeof(graph_node), cmp_edge_dst);

	int *anyChange, check_if_change;
	graph_node *L;
	unsigned int *hostDistance = (unsigned int *)malloc((sizeof(unsigned int))*(graph->size()));

	cudaMalloc((void**)&L, sizeof(graph_node)*edge_counter);
	cudaMalloc((void**)&distance, sizeof(unsigned int)*(graph->size()));
	cudaMalloc((void**)&anyChange, sizeof(int));

	cudaMemcpy(distance, initDist, sizeof(unsigned int)*(graph->size()), cudaMemcpyHostToDevice);
	cudaMemcpy(L, edge_list, sizeof(graph_node)*edge_counter, cudaMemcpyHostToDevice);
	
	cudaMemset(anyChange, 0, (size_t)sizeof(int));

    setTime();

	for(int i=0; i < (graph->size())-1; i++){
		edge_process_incore<<<blockNum,blockSize>>>(L, edge_counter, distance, anyChange);
		cudaMemcpy(&check_if_change, anyChange, sizeof(int), cudaMemcpyDeviceToHost);
		if(check_if_change == 0){
			break;
		} 
		cudaMemset(anyChange, 0, (size_t)sizeof(int));
	}

	std::cout << "Took " << getTime() << "ms.\n";

	cudaMemcpy(hostDistance, distance, (sizeof(unsigned int))*(graph->size()), cudaMemcpyDeviceToHost);

	for(int i=0; i < graph->size(); i++){
		if(hostDistance[i] == UINT_MAX){
		    outputFile << i << ":" << "INF" << endl;
		}
		else{
		    outputFile << i << ":" << hostDistance[i] << endl; 
		}
	}

	cudaFree(L);	
	cudaFree(distance);
	cudaFree(anyChange);
	
	delete[] hostDistance;
	free(initDist);
	free(edge_list);
}


//using shared memory
__device__ uint get_min(uint val1, uint val2) {
	if (val1 < val2) {
		return val1;
	} else {
		return val2;
	}
}

//smem kernel method
__global__ void edge_process_smem(const graph_node *L, const unsigned int edge_counter, unsigned int *distance_prev, unsigned int *distance_cur, int *anyChange){
	__shared__ int rows[1024];
	__shared__ int vals[1024];

	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	int total_threads = blockDim.x * gridDim.x;
	int warp_id = thread_id/32;
	int warp_num;
	if(total_threads % 32 == 0){
		warp_num = total_threads/32;
	}
	else{
		warp_num = total_threads/32 + 1;
	}
	int lane_id = thread_id % 32;
	
	int load = (edge_counter % warp_num == 0) ? edge_counter/warp_num : edge_counter/warp_num+1;
	int beg = load * warp_id;
	int end = beg + load;
	if(edge_counter < beg + load)
		end = edge_counter;
	beg = beg + lane_id;
	
	unsigned int u, v, w;
	unsigned int temp;

	for(int i = beg; i < end; i+=32){
		u = L[i].src;
		v = L[i].dst;
		w = L[i].weight;

		rows[threadIdx.x] = v;
		temp = distance_cur[v];
		
		if(distance_prev[u] == UINT_MAX){
		    vals[threadIdx.x] = UINT_MAX;
		} 
		else {
		    vals[threadIdx.x] = distance_prev[u] + w;
		}

		int lane = thread_id % 32;
		if (lane >= 1 && rows[threadIdx.x] == rows[threadIdx.x - 1])
			vals[threadIdx.x] = get_min(vals[threadIdx.x], vals[threadIdx.x - 1]);
		if (lane >= 2 && rows[threadIdx.x] == rows[threadIdx.x - 2])
			vals[threadIdx.x] = get_min(vals[threadIdx.x], vals[threadIdx.x - 2]);
		if (lane >= 4 && rows[threadIdx.x] == rows[threadIdx.x - 4])
			vals[threadIdx.x] = get_min(vals[threadIdx.x], vals[threadIdx.x - 4]);
		if (lane >= 8 && rows[threadIdx.x] == rows[threadIdx.x - 8])
			vals[threadIdx.x] = get_min(vals[threadIdx.x], vals[threadIdx.x - 8]);
		if (lane >= 16 && rows[threadIdx.x] == rows[threadIdx.x - 16])
			vals[threadIdx.x] = get_min(vals[threadIdx.x], vals[threadIdx.x - 16]);
		//write output if we are dealing with last thread in warp or rows are different
		if ((lane == 31) || (rows[threadIdx.x] != rows[threadIdx.x + 1])){
			atomicMin(&distance_cur[rows[threadIdx.x]], vals[threadIdx.x]);
		}
		
		if(distance_cur[v] < temp)
		    anyChange[0] = 1;
	}
}

////smem device method
void puller_smem(std::vector<initial_vertex> * graph, int blockSize, int blockNum, ofstream &outputFile){
	unsigned int *distance_cur, *distance_prev; 
    
    unsigned int *initDist;
	//set initial distance to max except for source node
	initDist = (unsigned int*)malloc(sizeof(unsigned int)*graph->size());	
	initDist[0] = 0;
	for(int i = 1; i < graph->size(); i++){
	    initDist[i] = UINT_MAX; 
	}
	
	unsigned int edge_counter;
	edge_counter = total_edges(*graph);
	graph_node *edge_list;
	edge_list = (graph_node*) malloc(sizeof(graph_node)*edge_counter);
	//for each member of edge list set initial values
	unsigned int k = 0;
	for(int i = 0 ; i < graph->size() ; i++){
		std::vector<neighbor> nbrs = (*graph)[i].nbrs;
	    for(int j = 0 ; j < nbrs.size() ; j++, k++){
			edge_list[k].src = nbrs[j].srcIndex;
			edge_list[k].dst = i;
			edge_list[k].weight = nbrs[j].edgeValue.weight;
	    }
	}
	//sort by dst
	//http://www.cplusplus.com/reference/cstdlib/qsort/
	qsort(edge_list, edge_counter, sizeof(graph_node), cmp_edge_dst);
	
	graph_node *L;			
	unsigned int *swapDistVariable = new unsigned int[graph->size()];
	int *anyChange, check_if_change;

	cudaMalloc((void**)&L, (size_t)sizeof(graph_node)*edge_counter);
	cudaMalloc((void**)&distance_cur, (size_t)sizeof(unsigned int)*(graph->size()));
	cudaMalloc((void**)&distance_prev, (size_t)sizeof(unsigned int)*(graph->size()));
	cudaMalloc((void**)&anyChange, (size_t)sizeof(int));

	cudaMemcpy(distance_cur, initDist, (size_t)sizeof(unsigned int)*(graph->size()), cudaMemcpyHostToDevice);
	cudaMemcpy(distance_prev, initDist, (size_t)sizeof(unsigned int)*(graph->size()), cudaMemcpyHostToDevice);
	cudaMemcpy(L, edge_list, (size_t)sizeof(graph_node)*edge_counter, cudaMemcpyHostToDevice);
	cudaMemset(anyChange, 0, (size_t)sizeof(int));

    setTime();

	for(int i=0; i < (graph->size())-1; i++){
		edge_process_smem<<<blockNum,blockSize>>>(L, edge_counter, distance_prev, distance_cur, anyChange);
		cudaMemcpy(&check_if_change, anyChange, sizeof(int), cudaMemcpyDeviceToHost);
		if(check_if_change == 0){
			break;
		} 
		cudaMemset(anyChange, 0, (size_t)sizeof(int));
		cudaMemcpy(swapDistVariable, distance_cur, (sizeof(unsigned int))*(graph->size()), cudaMemcpyDeviceToHost);
		cudaMemcpy(distance_cur, distance_prev, (sizeof(unsigned int))*(graph->size()), cudaMemcpyDeviceToDevice);
		cudaMemcpy(distance_prev, swapDistVariable,(sizeof(unsigned int))*(graph->size()), cudaMemcpyHostToDevice);
	}

	cout << "Took " << getTime() << "ms.\n";

	cudaMemcpy(swapDistVariable, distance_cur, (sizeof(unsigned int))*(graph->size()), cudaMemcpyDeviceToHost);

	for(int i=0; i < graph->size(); i++){
		if(swapDistVariable[i] == UINT_MAX){
		    outputFile << i << ":" << "INF" << endl;
		}
		else{
		    outputFile << i << ":" << swapDistVariable[i] << endl; 
		}
	}

	cudaFree(L);
	cudaFree(distance_cur);
	cudaFree(distance_prev);
	cudaFree(anyChange);
	
	delete[] swapDistVariable;
	free(initDist);
	free(edge_list);
}