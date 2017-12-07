#include <vector>
#include <iostream>
#include <thrust/scan.h>
#include "utils.h"
#include "cuda_error_check.cuh"
#include "initial_graph.hpp"
#include "parse_graph.hpp"

using namespace std;

//comparator function used by qsort
int cmp_edge_src(const void *a, const void *b){
	return ( (((graph_node *)a)->src) - (((graph_node *)b)->src));
}

//get total edges
unsigned int total_edges_opt(std::vector<initial_vertex>& graph){
	unsigned int edge_counter = 0;
	for(int i = 0 ; i < graph.size() ; i++){
	    edge_counter += graph[i].nbrs.size();
	}
	return edge_counter;
}

//outcore
//kernel outcore method
__global__ void edge_process_opt(graph_node *L, const uint edge_counter, unsigned int *distance_cur, unsigned int *distance_prev, unsigned int *queueCounter, unsigned int *nodeQueue){

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
	//printf("Hello thread %d\n", threadIdx.x);
	for(int i = beg; i < end; i+=32){
		edge = L + i;
		u = edge->src;
		v = edge->dst;
		w = edge->weight;
		if(distance_prev[u] != UINT_MAX && distance_prev[u] + w < distance_prev[v] && distance_prev[u] + w < distance_cur[v]){
			printf("Hello thread %d\n", threadIdx.x);
			int old_val = atomicMin(&distance_cur[v], distance_prev[u] + w);
			printf("old_val %d\n", old_val);
			if(old_val >= distance_prev[v] && distance_prev[u] + w < old_val){
				int idx = atomicAdd(&queueCounter[0], 1);
				printf("idx %d\n", idx);
				nodeQueue[idx] = v;
				printf("v %d\n", v);
			}
		}
	}
}

//device outcore method
void puller_outcore_impl3(std::vector<initial_vertex> * graph, int blockSize, int blockNum, ofstream &outputFile){
	double filter_time = 0;
	double compute_time = 0;

	unsigned int *initDist;
	//set initial distance to max except for source node
	initDist = (unsigned int*)malloc(sizeof(unsigned int)*graph->size());
	initDist[0] = 0;
	for(int i = 1; i < graph->size(); i++){
	    initDist[i] = UINT_MAX;
	}

	unsigned int edge_counter = total_edges_opt(*graph);
	unsigned int initial_edge_counter = edge_counter;

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
	qsort(edge_list, edge_counter, sizeof(graph_node), cmp_edge_src);

	//create allneighbor
	unsigned int *allNeighborNumber = new unsigned int[graph->size()];
	for(int i = 0; i < graph->size(); i++){
		allNeighborNumber[i] = 0;
	}
	for(int i = 0; i < graph->size(); i++){
	    std::vector<neighbor> nbrs = (*graph)[i].nbrs;
	    for(int j = 0 ; j < nbrs.size() ; j++){
	    	int src = nbrs[j].srcIndex;
	    	allNeighborNumber[src] += 1;
	    }
	}

  //create allOffsets
  unsigned int *allOffsets = new unsigned int[graph->size() + 1];
  for(int i = 0; i < graph->size(); i++){
	    allOffsets[i] = allNeighborNumber[i];
	}
  thrust::exclusive_scan(allOffsets, allOffsets + graph->size(), allOffsets);
  allOffsets[graph->size()] = allOffsets[graph->size() - 1] + allNeighborNumber[graph->size() - 1];

	std::cout << "before l_new" << "\n";
	//create a new l'
	graph_node *L_new;
	L_new = (graph_node*) malloc(sizeof(graph_node)*edge_counter);
	for(int i = 0, j=0 ; i < edge_counter; i++){
		if(edge_list[i].src == 0){
			L_new[j].src = edge_list[i].src;
			L_new[j].dst = edge_list[i].dst;
			L_new[j].weight = edge_list[i].weight;
			j++;
		}
	}

	std::cout << "before nodeQueue" << "\n";
	//create nodeQueue
	unsigned int *nodeQueue = (unsigned int*)malloc(sizeof(unsigned int)*graph->size());
	unsigned int *device_nodeQueue = (unsigned int*)malloc(sizeof(unsigned int)*graph->size());
	cudaMalloc((void**)&device_nodeQueue, sizeof(unsigned int)*graph->size());
	for(int i = 0; i < graph->size(); i++){
		nodeQueue[i] = -1;
	}
	cudaMemcpy(device_nodeQueue, nodeQueue, sizeof(unsigned int)*graph->size(), cudaMemcpyHostToDevice);

	std::cout << "before queueCounter" << "\n";
	//queueCounter
	unsigned int *queueCounter = 0;
	unsigned int *device_queueCounter;
	cudaMalloc((void**)&device_queueCounter, sizeof(unsigned int));
	cudaMemcpy(device_queueCounter, &queueCounter, sizeof(uint), cudaMemcpyHostToDevice);
	cudaMemset(device_queueCounter, 0, sizeof(unsigned int));

	unsigned int *distance_cur, *distance_prev, *temp_distance, *device_warp_update_ds, *device_edge_counter;
	graph_node *L;
	graph_node *device_L_new;

	int total_threads = blockSize * blockNum;
	int warp_num;
	if(total_threads % 32 == 0){
		warp_num = total_threads/32;
	}
	else{
		warp_num = total_threads/32 + 1;
	}

	std::cout << "before cuda thingys" << "\n";
	unsigned int *temp_warp_update = new unsigned int[warp_num];

	cudaMalloc((void**)&device_edge_counter, sizeof(unsigned int));
	cudaMalloc((void**)&temp_distance, sizeof(unsigned int)*graph->size());

	cudaMalloc((void**)&device_warp_update_ds, sizeof(unsigned int)*warp_num);
	cudaMalloc((void**)&L, sizeof(graph_node)*edge_counter);
	cudaMalloc((void**)&device_L_new, sizeof(graph_node)*edge_counter);
	cudaMalloc((void**)&distance_cur, sizeof(unsigned int)*graph->size());
	cudaMalloc((void**)&distance_prev, sizeof(unsigned int)*graph->size());

	cudaMemcpy(distance_cur, initDist, sizeof(unsigned int)*graph->size(), cudaMemcpyHostToDevice);
	cudaMemcpy(distance_prev, initDist, sizeof(unsigned int)*graph->size(), cudaMemcpyHostToDevice);
	cudaMemcpy(L, edge_list, sizeof(graph_node)*edge_counter, cudaMemcpyHostToDevice);
	cudaMemcpy(device_L_new, L_new, sizeof(graph_node)*edge_counter, cudaMemcpyHostToDevice);
	cudaMemcpy(device_edge_counter, &edge_counter, sizeof(uint), cudaMemcpyHostToDevice);

	for (int i = 0; i < graph->size()-1; ++i) {
		std::cout << "inside for" << "\n";
		setTime();

		cudaMemset(device_warp_update_ds, 0, sizeof(uint)*warp_num);
		cudaMemcpy(temp_distance, distance_cur, sizeof(uint)*graph->size(), cudaMemcpyDeviceToDevice);
		std::cout << "before kernel" << "\n";
		edge_process_opt<<<blockNum, blockSize>>>(device_L_new, initial_edge_counter, distance_cur, distance_prev, device_queueCounter, device_nodeQueue);
		cudaDeviceSynchronize();
		std::cout << "after kernel"<< "\n";
		cudaMemcpy(distance_prev, distance_cur, sizeof(uint)*graph->size(), cudaMemcpyDeviceToDevice);
		cudaMemcpy(nodeQueue, device_nodeQueue, sizeof(uint)*graph->size(), cudaMemcpyDeviceToHost);

		for(int j = 0; j < 10; j++){
			std::cout << nodeQueue[j] << "\n";
		}
		break;
		compute_time += getTime();

		setTime();

		filter_time += getTime();
	}

	std::cout << "Compute Time: " << compute_time << "\n";
	std::cout << "Filter Time: " << filter_time;

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
	cudaFree(L);
	cudaFree(device_edge_counter);
	cudaFree(distance_cur);
	cudaFree(distance_prev);
	cudaFree(temp_distance);
	cudaFree(device_warp_update_ds);
}
