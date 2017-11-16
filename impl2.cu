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

__global__ void prefix_sum(uint* warp_count, uint warp_num, uint *nEdges) {

	__shared__ uint shared_mem[2048];

	int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	int threadCount = blockDim.x * gridDim.x;
	if (threadId == 0) {
		*nEdges = warp_count[warp_num - 1];
	}
	
	if (threadId < warp_num)
		shared_mem[threadId] = warp_count[threadId];

	__syncthreads();

	for (int offset = 1; offset < warp_num; offset *= 2) {
		for (int idx = threadId; idx < warp_num; idx += threadCount) {
			if (idx >= offset) {
				warp_count[idx] += warp_count[idx - offset];
			}
		}

		__syncthreads();
	}	
	
	if (threadId == 0) {
		*nEdges += warp_count[warp_num - 1];
	}

	
	if (threadId < warp_num)
		warp_count[threadId] -= shared_mem[threadId];
}

//L, device_warp_update_ds, edge_counter, flag
__global__ void warp_count_kernel(graph_node *L, unsigned int *warp_update_ds, const unsigned int edge_counter, unsigned int *flag){
    
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
    graph_node *edge;
    for(int i = beg; i < end; i+=32){
    	edge = L + i;
		int mask = __ballot(flag[edge->src]);
	    temp_num = __popc(mask);
	    warp_update_ds[warp_id] += temp_num;
    }
}

__global__ void filter_kernel(graph_node *L, uint *edge_idx, unsigned int *warp_update_ds, const unsigned int edge_counter, unsigned int *flag){
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

	uint cur_offset = 0;
    graph_node *edge;
    for(int i = beg; i < end; i+=32){
    	edge = L + i;
		int mask = __ballot(flag[L[i].src]);
		int inner_idx = __popc(mask << (32 - lane_id));
		if(flag[edge->src]){
		    edge_idx[cur_offset + inner_idx + warp_update_ds[warp_id]]= i;
		}
		cur_offset += __popc(mask);
    }

}


//outcore
//kernel outcore method
__global__ void edge_process(graph_node *L, uint *edge_idx, const uint edge_counter, unsigned int *distance_cur, unsigned int *distance_prev, int *anyChange, unsigned int *flag){
	
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
	graph_node *edge;
	for(int i = beg; i < end; i+=32){
		edge = L + edge_idx[i];
		u = edge->src;
		v = edge->dst;
		w = edge->weight;
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

	graph_node *edge_list, *L;
	
	unsigned int *distance_cur, *distance_prev, *flag, *temp_distance, *device_warp_update_ds, *device_edge_counter, *edge_offset_ds;
	
	int *anyChange;
	int check_if_change;

	unsigned int edge_counter = total_edges_impl2(*graph);
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

	unsigned int initial_edge_counter = edge_counter;

	//todo h_edge_idx
	unsigned int *h_edge_idx = new unsigned int[edge_counter];
	for (int i = 0; i < edge_counter; ++i) {
		h_edge_idx[i] = i;
	}

	int total_threads = blockSize * blockNum;
	int warp_num;
	if(total_threads % 32 == 0){
		warp_num = total_threads/32;
	}
	else{
		warp_num = total_threads/32 + 1;
	}
	
	unsigned int *initDist;
	//set initial distance to max except for source node
	initDist = (unsigned int*)malloc(sizeof(unsigned int)*graph->size());	
	initDist[0] = 0;
	for(int i = 1; i < graph->size(); i++){
	    initDist[i] = UINT_MAX; 
	}

	
	cudaMalloc((void**)&edge_offset_ds, sizeof(unsigned int)*edge_counter);
	cudaMalloc((void**)&device_edge_counter, sizeof(unsigned int));
	cudaMalloc((void**)&temp_distance, sizeof(unsigned int)*graph->size());	

	cudaMalloc((void**)&device_warp_update_ds, sizeof(unsigned int)*warp_num);
	cudaMalloc((void**)&flag, sizeof(unsigned int)*graph->size());
	cudaMalloc((void**)&L, sizeof(graph_node)*edge_counter);
	cudaMalloc((void**)&distance_cur, sizeof(unsigned int)*graph->size());
	cudaMalloc((void**)&distance_prev, sizeof(unsigned int)*graph->size());
	cudaMalloc((void**)&anyChange, sizeof(int));
	
	cudaMemcpy(distance_cur, initDist, sizeof(unsigned int)*graph->size(), cudaMemcpyHostToDevice);
	cudaMemcpy(distance_prev, initDist, sizeof(unsigned int)*graph->size(), cudaMemcpyHostToDevice);
	cudaMemcpy(L, edge_list, sizeof(graph_node)*edge_counter, cudaMemcpyHostToDevice);
	cudaMemcpy(device_edge_counter, &edge_counter, sizeof(uint), cudaMemcpyHostToDevice);
	cudaMemcpy(edge_offset_ds, h_edge_idx, sizeof(uint)*edge_counter, cudaMemcpyHostToDevice);

	for (int i = 0; i < graph->size()-1; ++i) {
		setTime();
	
		cudaMemset(anyChange, 0, sizeof(int));
		cudaMemset(flag, 0, sizeof(uint)*graph->size());
		cudaMemset(device_warp_update_ds, 0, sizeof(uint)*warp_num);
		cudaMemcpy(temp_distance, distance_cur, sizeof(uint)*graph->size(), cudaMemcpyDeviceToDevice);

		edge_process<<<blockNum, blockSize>>>(L, edge_offset_ds, initial_edge_counter, distance_cur, distance_prev, anyChange, flag);
		cudaDeviceSynchronize();
		cudaMemcpy(distance_prev, distance_cur, sizeof(uint)*graph->size(), cudaMemcpyDeviceToDevice);
		cudaMemcpy(&check_if_change, anyChange, sizeof(int), cudaMemcpyDeviceToHost);
		
		compute_time += getTime();
		
		if (check_if_change == 0) {
			break;
		}

		setTime();
		
		warp_count_kernel<<<blockNum, blockSize>>>(L, device_warp_update_ds, edge_counter, flag);		
		cudaDeviceSynchronize();

		prefix_sum<<<blockNum, blockSize>>>(device_warp_update_ds, warp_num, device_edge_counter);
		cudaDeviceSynchronize();
		
		filter_kernel<<<blockNum, blockSize>>>(L, edge_offset_ds, device_warp_update_ds, edge_counter, flag);
		cudaDeviceSynchronize();
		
		cudaMemcpy(&initial_edge_counter, device_edge_counter, sizeof(uint), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_edge_idx, edge_offset_ds, sizeof(uint)*edge_counter, cudaMemcpyDeviceToHost);
		
		filter_time += getTime();
	}

	std::cout << compute_time + filter_time << "ms.(filter - "<<filter_time<<"ms processing - "<<compute_time<<"ms)\n";

	cudaMemcpy(initDist, distance_cur, sizeof(uint)*graph->size(), cudaMemcpyDeviceToHost);

	for(int i=0; i < graph->size(); i++){
		if(initDist[i] == UINT_MAX){
		    outputFile << i << ":" << "INF" << endl;
		}
		else{
		    outputFile << i << ":" << initDist[i] << endl; 
		}
	}
	free(initDist);
	delete[] h_edge_idx;
	cudaFree(L);
	cudaFree(edge_offset_ds);
	cudaFree(device_edge_counter);
	cudaFree(flag);
	cudaFree(distance_cur);
	cudaFree(distance_prev);
	cudaFree(temp_distance);
	cudaFree(device_warp_update_ds);
	cudaFree(anyChange);
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
		    //set_wrap_count<<<blockNum, blockSize>>>(L, edge_counter, flag, warp_update_ds);
		    cudaDeviceSynchronize();
		    cudaMemcpy(temp, warp_update_ds + warp_num - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
		    to_process_num = *temp;
		    cudaMemcpy(device_warp_update_ds, warp_update_ds, sizeof(unsigned int)*warp_num, cudaMemcpyDeviceToHost);
		    thrust::exclusive_scan(device_warp_update_ds, device_warp_update_ds + warp_num, device_warp_update_ds);
		    cudaMemcpy(warp_update_ds, device_warp_update_ds, sizeof(unsigned int)*warp_num, cudaMemcpyHostToDevice);
		    cudaMemcpy(temp, warp_update_ds + warp_num - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
		    to_process_num += *temp;
		    cudaMalloc((void**)&T, (size_t)sizeof(graph_node)*to_process_num);
		    //filter_T<<<blockNum, blockSize>>>(L, edge_counter, flag, warp_update_ds, T);
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