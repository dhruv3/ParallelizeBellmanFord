#include <vector>
#include <iostream>
#include <thrust/scan.h>
#include "utils.h"
#include "cuda_error_check.cuh"
#include "initial_graph.hpp"
#include "parse_graph.hpp"

using namespace std;

//comparator function used by qsort
int cmp_edge1(const void *a, const void *b){	
	return ( (int)(((graph_node *)a)->src) - (int)(((graph_node *)b)->src));
}

//get total edges
unsigned int total_edges_impl2(std::vector<initial_vertex>& graph){
	unsigned int edge_counter = 0;
	for(int i = 0 ; i < graph.size() ; i++){
	    edge_counter += graph[i].nbrs.size();
	}
	return edge_counter;
}

__global__ void set_wrap_count(const graph_node *L, const unsigned int edge_counter, unsigned int *flag, unsigned int *warp_update_ds){
    
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

    unsigned int temp_num = 0;
    for(int i = beg; i < end; i+=32){
		int mask = __ballot(flag[L[i].src]);
		if(lane_id == 0){
		    temp_num += (unsigned int) __popc(mask);
		}
    }

    if(lane_id == 0){
		warp_update_ds[warp_id] = temp_num;
    }
}

__global__ void filter_T(const graph_node *L, const unsigned int edge_counter, unsigned int *flag, unsigned int *warp_update_ds, graph_node *T){
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

	int cur_offset = warp_update_ds[warp_id];
    
    for(int i = beg; i < end; i+=32){
		int mask = __ballot(flag[L[i].src]);
		int inner_idx = __popc(mask << (32 - 1) - lane_id) - 1;
		if(flag[L[i].src]){
		    T[cur_offset+inner_idx]= L[i];
		}
		cur_offset += __popc(mask);
    }

}


//outcore
//kernel outcore method
__global__ void edge_process(const graph_node *L, const unsigned int edge_counter, unsigned int *distance_prev, unsigned int *distance_cur, int *anyChange, unsigned int *flag){
	
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
			flag[v] = 1;
		}
	}
}

//device outcore method
void puller_outcore_impl2(std::vector<initial_vertex> * graph, int blockSize, int blockNum, ofstream &outputFile){
	double filter_time = 0;
	double compute_time = 0;

	unsigned int *initDist, *distance_cur, *distance_prev, *warp_update_ds, *flag; 
	int *anyChange;
	int *checkIfChange = (int*)malloc(sizeof(int));
	graph_node *edge_list, *L, *T;
	unsigned int edge_counter, to_process_num;
	edge_counter = total_edges_impl2(*graph);
	edge_list = (graph_node*) malloc(sizeof(graph_node)*edge_counter);
	
	unsigned int *temp = (unsigned int*)malloc(sizeof(unsigned int));
	int total_threads = blockSize * blockNum;
	int warp_num;
	if(total_threads % 32 == 0){
		warp_num = total_threads/32;
	}
	else{
		warp_num = total_threads/32 + 1;
	}


	//set initial distance to max except for source node
	initDist = (unsigned int*)malloc(sizeof(unsigned int)*graph->size());	
	initDist[0] = 0;
	for(int i = 1; i < graph->size(); i++){
	    initDist[i] = UINT_MAX; 
	}
	
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
	qsort(edge_list, edge_counter, sizeof(graph_node), cmp_edge1);			

	unsigned int *swapDistVariable = new unsigned int[graph->size()];
	unsigned int *device_warp_update_ds = new unsigned int[warp_num];

	cudaMalloc((void**)&warp_update_ds, (size_t)sizeof(unsigned int)*warp_num);
	cudaMalloc((void**)&flag, (size_t)sizeof(unsigned int)*(graph->size()));
	cudaMalloc((void**)&L, (size_t)sizeof(graph_node)*edge_counter);
	cudaMalloc((void**)&distance_cur, (size_t)sizeof(unsigned int)*(graph->size()));
	cudaMalloc((void**)&distance_prev, (size_t)sizeof(unsigned int)*(graph->size()));
	cudaMalloc((void**)&anyChange, (size_t)sizeof(int));
	

	cudaMemcpy(distance_cur, initDist, (size_t)sizeof(unsigned int)*(graph->size()), cudaMemcpyHostToDevice);
	cudaMemcpy(distance_prev, initDist, (size_t)sizeof(unsigned int)*(graph->size()), cudaMemcpyHostToDevice);
	cudaMemcpy(L, edge_list, (size_t)sizeof(graph_node)*edge_counter, cudaMemcpyHostToDevice);
	cudaMemset(warp_update_ds, 0, (size_t)sizeof(unsigned int)*warp_num);
	cudaMemset(flag, 0, (size_t)sizeof(unsigned int)*(graph->size()));
	cudaMemset(anyChange, 0, (size_t)sizeof(int));

	for(int i=0; i < ((int) graph->size())-1; i++){
		setTime();
		if(i == 0){
		    edge_process<<<blockNum,blockSize>>>(L, edge_counter, distance_prev, distance_cur, anyChange, flag);
		    cudaDeviceSynchronize();
		} 
		else {
		    cudaMemset(flag, 0, (size_t)sizeof(unsigned int)*(graph->size()));
		    edge_process<<<blockNum,blockSize>>>(L, edge_counter, distance_prev, distance_cur, anyChange, flag);
		    cudaDeviceSynchronize();
		    cudaFree(T);
		}
		compute_time += getTime();
		cudaMemcpy(checkIfChange, anyChange, sizeof(int), cudaMemcpyDeviceToHost);
		if(!checkIfChange[0]){
			break;
		} 
		else {
			cudaMemset(anyChange, 0, (size_t)sizeof(int));
			cudaMemcpy(distance_prev, distance_cur, (sizeof(unsigned int))*(graph->size()), cudaMemcpyDeviceToDevice);
			cudaMemcpy(swapDistVariable, distance_cur, (sizeof(unsigned int))*(graph->size()), cudaMemcpyDeviceToHost);
		}

		if(i == graph->size() - 2){
		    break;
		}
		else{
		    setTime();
		    cudaMemset(warp_update_ds, 0, (size_t)sizeof(unsigned int)*warp_num);
		    set_wrap_count<<<blockNum, blockSize>>>(L, edge_counter, flag, warp_update_ds);
		    cudaDeviceSynchronize();
		    cudaMemcpy(temp, warp_update_ds + warp_num - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
		    to_process_num = *temp;
		    cudaMemcpy(device_warp_update_ds, warp_update_ds, sizeof(unsigned int)*warp_num, cudaMemcpyDeviceToHost);
		    thrust::exclusive_scan(device_warp_update_ds, device_warp_update_ds + warp_num, device_warp_update_ds);
		    cudaMemcpy(warp_update_ds, device_warp_update_ds, sizeof(unsigned int)*warp_num, cudaMemcpyHostToDevice);
		    cudaMemcpy(temp, warp_update_ds + warp_num - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
		    to_process_num += *temp;
		    cudaMalloc((void**)&T, (size_t)sizeof(graph_node)*to_process_num);
		    filter_T<<<blockNum, blockSize>>>(L, edge_counter, flag, warp_update_ds, T);
		    cudaDeviceSynchronize();
		    filter_time += getTime();
		}
	}

	printf("Computation Time: %f ms\nFiltering Time: %f ms\n", compute_time, filter_time);

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

	delete[] device_warp_update_ds;
	delete[] swapDistVariable;
	free(initDist);
	free(edge_list);
}

//incore
//kernel incore method
__global__ void edge_process_incore(const graph_node *L, const unsigned int edge_counter, unsigned int *distance, int *anyChange, unsigned int *flag){
	
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
		if(distance[u] != UINT_MAX && distance[u] + w < distance[v]){
			atomicMin(&distance[v], distance[u] + w);
			anyChange[0] = 1;
			flag[v] = 1;
		}
	}
}

//device incore method
void puller_incore_impl2(std::vector<initial_vertex> * graph, int blockSize, int blockNum, ofstream &outputFile){
	double filter_time = 0;
	double compute_time = 0;

	unsigned int *initDist, *distance, *warp_update_ds, *flag; 
	int *anyChange;
	int *checkIfChange = (int*)malloc(sizeof(int));
	graph_node *edge_list, *L, *T;
	unsigned int edge_counter, to_process_num;
	edge_counter = total_edges_impl2(*graph);
	edge_list = (graph_node*) malloc(sizeof(graph_node)*edge_counter);
	
	unsigned int *temp = (unsigned int*)malloc(sizeof(unsigned int));
	int total_threads = blockSize * blockNum;
	int warp_num;
	if(total_threads % 32 == 0){
		warp_num = total_threads/32;
	}
	else{
		warp_num = total_threads/32 + 1;
	}


	//set initial distance to max except for source node
	initDist = (unsigned int*)malloc(sizeof(unsigned int)*graph->size());	
	initDist[0] = 0;
	for(int i = 1; i < graph->size(); i++){
	    initDist[i] = UINT_MAX; 
	}
	
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
	qsort(edge_list, edge_counter, sizeof(graph_node), cmp_edge1);			

	unsigned int *swapDistVariable = new unsigned int[graph->size()];
	unsigned int *device_warp_update_ds = new unsigned int[warp_num];

	cudaMalloc((void**)&warp_update_ds, (size_t)sizeof(unsigned int) * warp_num);
	cudaMalloc((void**)&flag, (size_t)sizeof(unsigned int) * (graph->size()));
	cudaMalloc((void**)&L, (size_t)sizeof(graph_node)*edge_counter);
	cudaMalloc((void**)&distance, (size_t)sizeof(unsigned int)*(graph->size()));
	cudaMalloc((void**)&anyChange, (size_t)sizeof(int));
	
	cudaMemcpy(distance, initDist, (size_t)sizeof(unsigned int)*(graph->size()), cudaMemcpyHostToDevice);
	cudaMemcpy(L, edge_list, (size_t)sizeof(graph_node)*edge_counter, cudaMemcpyHostToDevice);
	cudaMemset(warp_update_ds, 0, (size_t)sizeof(unsigned int)*warp_num);
	cudaMemset(flag, 0, (size_t)sizeof(unsigned int)*(graph->size()));
	cudaMemset(anyChange, 0, (size_t)sizeof(int));

	for(int i=0; i < ((int) graph->size())-1; i++){
		setTime();
		if(i == 0){
		    edge_process_incore<<<blockNum,blockSize>>>(L, edge_counter, distance, anyChange, flag);
		} 
		else {
		    cudaMemset(flag, 0, (size_t)sizeof(unsigned int)*(graph->size()));
		    edge_process_incore<<<blockNum,blockSize>>>(L, edge_counter, distance, anyChange, flag);
		    cudaFree(T);
		}
		compute_time += getTime();
		cudaMemcpy(checkIfChange, anyChange, sizeof(int), cudaMemcpyDeviceToHost);
		if(!checkIfChange[0]){
			break;
		} 
		else {
			cudaMemset(anyChange, 0, (size_t)sizeof(int));
		}

		if(i == graph->size() - 2){
		    break;
		}
		else{
		    setTime();
		    cudaMemset(warp_update_ds, 0, (size_t)sizeof(unsigned int)*warp_num);
		    set_wrap_count<<<blockNum, blockSize>>>(L, edge_counter, flag, warp_update_ds);
		    cudaDeviceSynchronize();
		    cudaMemcpy(temp, warp_update_ds + warp_num - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
		    to_process_num = *temp;
		    cudaMemcpy(device_warp_update_ds, warp_update_ds, sizeof(unsigned int)*warp_num, cudaMemcpyDeviceToHost);
		    thrust::exclusive_scan(device_warp_update_ds, device_warp_update_ds + warp_num, device_warp_update_ds);
		    cudaMemcpy(warp_update_ds, device_warp_update_ds, sizeof(unsigned int)*warp_num, cudaMemcpyHostToDevice);
		    cudaMemcpy(temp, warp_update_ds + warp_num - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
		    to_process_num += *temp;
		    cudaMalloc((void**)&T, (size_t)sizeof(graph_node)*to_process_num);
		    filter_T<<<blockNum, blockSize>>>(L, edge_counter, flag, warp_update_ds, T);
		    cudaDeviceSynchronize();
		    filter_time += getTime();
		}
	}

	printf("Computation Time: %f ms\nFiltering Time: %f ms\n", compute_time, filter_time);

	cudaMemcpy(swapDistVariable, distance, (sizeof(unsigned int))*(graph->size()), cudaMemcpyDeviceToHost);
	for(int i=0; i < graph->size(); i++){
		if(swapDistVariable[i] == UINT_MAX){
		    outputFile << i << ":" << "INF" << endl;
		}
		else{
		    outputFile << i << ":" << swapDistVariable[i] << endl; 
		}
	}

	cudaFree(L);
	cudaFree(distance);
	cudaFree(anyChange);

	delete[] device_warp_update_ds;
	delete[] swapDistVariable;
	free(initDist);
	free(edge_list);
}