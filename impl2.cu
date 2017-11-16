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

__global__ void prefix_sum(uint* warp_count, uint count_warps, uint *nEdges) {

	__shared__ uint shared_mem[2048];

	int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	int threadCount = blockDim.x * gridDim.x;
	if (threadId == 0) {
		*nEdges = warp_count[count_warps - 1];
	}
	
	if (threadId < count_warps)
		shared_mem[threadId] = warp_count[threadId];

	__syncthreads();

	for (int offset = 1; offset < count_warps; offset *= 2) {
		for (int idx = threadId; idx < count_warps; idx += threadCount) {
			if (idx >= offset) {
				warp_count[idx] += warp_count[idx - offset];
			}
		}

		__syncthreads();
	}	
	
	if (threadId == 0) {
		*nEdges += warp_count[count_warps - 1];
	}

	
	if (threadId < count_warps)
		warp_count[threadId] -= shared_mem[threadId];
}

//d_edges, d_warp_count, edge_counter, d_change
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
	graph_node *d_edges;
	uint* d_edge_idx;
	uint* d_distances_curr;
	uint* d_distances_prev;
	uint* d_distances_dummy;
	uint* d_warp_count;
	uint* d_nEdges;
	uint* d_change;
	int *d_is_changed;
	int h_is_changed;

	unsigned int edge_counter = total_edges_impl2(*graph);
	uint count_to_process = edge_counter;
	uint* h_edge_idx = new uint[edge_counter];
	int count_iterations = 0;
	int total_threads = blockSize * blockNum;
	int count_warps;
	if(total_threads % 32 == 0){
		count_warps = total_threads/32;
	}
	else{
		count_warps = total_threads/32 + 1;
	}
	unsigned int *initDist;
	//set initial distance to max except for source node
	initDist = (unsigned int*)malloc(sizeof(unsigned int)*graph->size());	
	initDist[0] = 0;
	for(int i = 1; i < graph->size(); i++){
	    initDist[i] = UINT_MAX; 
	}

	cudaMalloc((void**)&d_edges, sizeof(graph_node)*edge_counter);
	cudaMalloc((void**)&d_edge_idx, sizeof(uint)*edge_counter);
	cudaMalloc((void**)&d_nEdges, sizeof(uint));
	cudaMalloc((void**)&d_distances_curr, sizeof(uint)*graph->size());
	cudaMalloc((void**)&d_distances_prev, sizeof(uint)*graph->size());
	cudaMalloc((void**)&d_distances_dummy, sizeof(uint)*graph->size());
	cudaMalloc((void**)&d_change, sizeof(uint)*graph->size());
	cudaMalloc((void**)&d_warp_count, sizeof(uint)*count_warps);
	cudaMalloc((void**)&d_is_changed, sizeof(int));

	cudaMemcpy(d_edges, graph, sizeof(graph_node)*edge_counter, cudaMemcpyHostToDevice);
	cudaMemcpy(d_nEdges, &edge_counter, sizeof(uint), cudaMemcpyHostToDevice);
	cudaMemcpy(d_distances_curr, initDist, sizeof(uint)*graph->size(), cudaMemcpyHostToDevice);
	cudaMemcpy(d_distances_prev, initDist, sizeof(uint)*graph->size(), cudaMemcpyHostToDevice);

	for (int i = 0; i < edge_counter; ++i) {
		h_edge_idx[i] = i;
	}

	cudaMemcpy(d_edge_idx, h_edge_idx, sizeof(uint)*edge_counter, cudaMemcpyHostToDevice);
	
	double filter_time = 0.0;
	double processing_time = 0.0;

	for (int i = 0; i < graph->size()-1; ++i) {
		setTime();
	
		cudaMemset(d_is_changed, 0, sizeof(int));
		cudaMemset(d_change, 0, sizeof(uint)*graph->size());
		cudaMemset(d_warp_count, 0, sizeof(uint)*count_warps);
		cudaMemcpy(d_distances_dummy, d_distances_curr, sizeof(uint)*graph->size(), cudaMemcpyDeviceToDevice);

		edge_process<<<blockNum, blockSize>>>(d_edges, d_edge_idx, count_to_process, d_distances_curr, d_distances_prev, d_is_changed, d_change);
		
		cudaDeviceSynchronize();

		cudaMemcpy(d_distances_prev, d_distances_curr, sizeof(uint)*graph->size(), cudaMemcpyDeviceToDevice);

		count_iterations++;

		cudaMemcpy(&h_is_changed, d_is_changed, sizeof(int), cudaMemcpyDeviceToHost);
		
		processing_time += getTime();
		
		if (h_is_changed == 0) {
			break;
		}

		setTime();
		warp_count_kernel<<<blockNum, blockSize>>>(d_edges, d_warp_count, edge_counter, d_change);		
		cudaDeviceSynchronize();
		prefix_sum<<<blockNum, blockSize>>>(d_warp_count, count_warps, d_nEdges);
		cudaDeviceSynchronize();
		filter_kernel<<<blockNum, blockSize>>>(d_edges, d_edge_idx, d_warp_count, edge_counter, d_change);
		cudaDeviceSynchronize();
		cudaMemcpy(&count_to_process, d_nEdges, sizeof(uint), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_edge_idx, d_edge_idx, sizeof(uint)*edge_counter, cudaMemcpyDeviceToHost);
		filter_time += getTime();
	}

	std::cout << "Took "<<count_iterations << " iterations " << processing_time + filter_time << "ms.(filter - "<<filter_time<<"ms processing - "<<processing_time<<"ms)\n";

	cudaMemcpy(initDist, d_distances_curr, sizeof(uint)*graph->size(), cudaMemcpyDeviceToHost);

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
	cudaFree(d_edges);
	cudaFree(d_edge_idx);
	cudaFree(d_nEdges);
	cudaFree(d_change);
	cudaFree(d_distances_curr);
	cudaFree(d_distances_prev);
	cudaFree(d_distances_dummy);
	cudaFree(d_warp_count);
	cudaFree(d_is_changed);
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