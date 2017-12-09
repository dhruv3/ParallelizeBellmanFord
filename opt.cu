#include <vector>
#include <iostream>
#include <thrust/scan.h>
#include <algorithm>
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
	for(int i = beg; i < end; i+=32){
		edge = L + i;
		u = edge->src;
		v = edge->dst;
		w = edge->weight;
		if(distance_prev[u] != UINT_MAX && distance_prev[u] + w < distance_prev[v] && distance_prev[u] + w < distance_cur[v]){
			int old_val = atomicMin(&distance_cur[v], distance_prev[u] + w);
			if(old_val >= distance_prev[v] && distance_prev[u] + w < old_val){
				int idx = atomicAdd(&queueCounter[0], 1);
				nodeQueue[idx] = v;
			}
		}
	}
}

__global__ void tpe_update(graph_node *tpe, unsigned int *nodeQueue, unsigned int *nodeOffsets, unsigned int *allOffsets, unsigned int *queueCounter, graph_node *edge_list){
	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	int total_threads = blockDim.x * gridDim.x;
	//listing 2
	int total_edges =  nodeOffsets[queueCounter[0]];
	int loadPerThread;
	if(total_edges % total_threads == 0){
		loadPerThread = nodeOffsets[queueCounter[0]] / total_threads;
	}
	else{
		loadPerThread = nodeOffsets[queueCounter[0]] / total_threads + 1;
	}
	unsigned int tid_begin = thread_id * loadPerThread;
	unsigned int tid_end;
	if(((thread_id + 1)*loadPerThread - 1) > (nodeOffsets[queueCounter[0]] - 1)){
		tid_end = nodeOffsets[queueCounter[0]] - 1;
	}
	else{
		tid_end = (thread_id + 1)*loadPerThread - 1;
	}
	//listing 3
	int i = 0;
	for(int t = 0; t < queueCounter[0]+1; t++){
		if(nodeOffsets[t] > tid_begin){
			if(t != 0){
				i = t-1;
			}
			break;
		}

		if(t == queueCounter[0]){
			i = t;
		}
	}

	int startVertex = nodeQueue[i];
	if(startVertex != -1){
		for(int j = tid_begin; j <= tid_end; j++){
			int phi = j - nodeOffsets[i];
			if(j < nodeOffsets[i+1]){
				uint edgePos = allOffsets[startVertex] + phi;
				tpe[j].src = edge_list[edgePos].src;
				tpe[j].dst = edge_list[edgePos].dst;
				tpe[j].weight = edge_list[edgePos].weight;
			}
			else{
				i++;
				if(i >= queueCounter[0]){
					printf("%d\n", i);
					return;
				}
				startVertex = nodeQueue[i];
				phi = j - nodeOffsets[i];
				uint edgePos = allOffsets[startVertex] + phi;
				tpe[j].src = edge_list[edgePos].src;
				tpe[j].dst = edge_list[edgePos].dst;
				tpe[j].weight = edge_list[edgePos].weight;
			}
			phi++;
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
	graph_node *device_edge_list;
	cudaMalloc((void**)&device_edge_list, sizeof(graph_node)*edge_counter);
	cudaMemcpy(device_edge_list, edge_list, sizeof(graph_node)*edge_counter, cudaMemcpyHostToDevice);

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
	unsigned int *device_allOffsets = new unsigned int[graph->size() + 1];
	cudaMalloc((void**)&device_allOffsets, sizeof(unsigned int)*(graph->size()+1));
	cudaMemcpy(device_allOffsets, allOffsets, sizeof(unsigned int)*(graph->size()+1), cudaMemcpyHostToDevice);

	//create a new tpe
	graph_node *tpe;
	tpe = (graph_node*) malloc(sizeof(graph_node)*edge_counter);
	for(int i = 0, j=0 ; i < edge_counter; i++){
		if(edge_list[i].src == 0){
			tpe[j].src = edge_list[i].src;
			tpe[j].dst = edge_list[i].dst;
			tpe[j].weight = edge_list[i].weight;
			j++;
		}
	}

	//create nodeQueue
	unsigned int *nodeQueue = (unsigned int*)malloc(sizeof(unsigned int)*graph->size());
	unsigned int *device_nodeQueue = (unsigned int*)malloc(sizeof(unsigned int)*graph->size());
	cudaMalloc((void**)&device_nodeQueue, sizeof(unsigned int)*graph->size());
	for(int i = 0; i < graph->size(); i++){
		nodeQueue[i] = -1;
	}
	cudaMemcpy(device_nodeQueue, nodeQueue, sizeof(unsigned int)*graph->size(), cudaMemcpyHostToDevice);

	//queueCounter
	unsigned int queueCounter = 0;
	unsigned int *device_queueCounter;
	cudaMalloc((void**)&device_queueCounter, sizeof(unsigned int));
	cudaMemcpy(device_queueCounter, &queueCounter, sizeof(uint), cudaMemcpyHostToDevice);
	cudaMemset(device_queueCounter, 0, sizeof(unsigned int));

	unsigned int *distance_cur, *distance_prev;
	graph_node *device_tpe;

	cudaMalloc((void**)&device_tpe, sizeof(graph_node)*edge_counter);
	cudaMalloc((void**)&distance_cur, sizeof(unsigned int)*graph->size());
	cudaMalloc((void**)&distance_prev, sizeof(unsigned int)*graph->size());

	cudaMemcpy(distance_cur, initDist, sizeof(unsigned int)*graph->size(), cudaMemcpyHostToDevice);
	cudaMemcpy(distance_prev, initDist, sizeof(unsigned int)*graph->size(), cudaMemcpyHostToDevice);
	cudaMemcpy(device_tpe, tpe, sizeof(graph_node)*edge_counter, cudaMemcpyHostToDevice);

	for (int i = 0; i < graph->size()-1; ++i) {
		edge_process_opt<<<blockNum, blockSize>>>(device_tpe, initial_edge_counter, distance_cur, distance_prev, device_queueCounter, device_nodeQueue);
		cudaDeviceSynchronize();
		cudaMemcpy(distance_prev, distance_cur, sizeof(uint)*graph->size(), cudaMemcpyDeviceToDevice);
		cudaMemcpy(nodeQueue, device_nodeQueue, sizeof(uint)*graph->size(), cudaMemcpyDeviceToHost);
		cudaMemcpy(&queueCounter, device_queueCounter, sizeof(uint), cudaMemcpyDeviceToHost);
		printf("new %d\n",queueCounter);
		if(queueCounter == 0){
			break;
		}
		//create neighborNumber
		unsigned int *neighborNumber = new unsigned int[queueCounter];
		for(int j = 0; j < queueCounter; j++){
			int idx = nodeQueue[j];
			neighborNumber[j] = allNeighborNumber[idx];
		}
		//create nodeOffset
		unsigned int *nodeOffsets = new unsigned int[queueCounter + 1];
		for(int i = 0; i < queueCounter; i++){
		    nodeOffsets[i] = neighborNumber[i];
		}
		thrust::exclusive_scan(nodeOffsets, nodeOffsets + queueCounter, nodeOffsets);
		nodeOffsets[queueCounter] = nodeOffsets[queueCounter - 1] + neighborNumber[queueCounter - 1];
		unsigned int *device_nodeOffsets = new unsigned int[queueCounter + 1];
		cudaMalloc((void**)&device_nodeOffsets, sizeof(uint)*(queueCounter + 1));
		cudaMemcpy(device_nodeOffsets, nodeOffsets, sizeof(uint)*(queueCounter + 1), cudaMemcpyHostToDevice);

		free(tpe);
		cudaFree(device_tpe);
		graph_node *tpe = (graph_node*) malloc(sizeof(graph_node)*edge_counter);
		graph_node *device_tpe;
		cudaMalloc((void**)&device_tpe, sizeof(graph_node)*edge_counter);
		cudaMemcpy(device_tpe, tpe, sizeof(graph_node)*edge_counter, cudaMemcpyHostToDevice);

		tpe_update<<<blockNum, blockSize>>>(device_tpe, device_nodeQueue, device_nodeOffsets, device_allOffsets, device_queueCounter, device_edge_list);
		cudaDeviceSynchronize();
		cudaMemcpy(tpe, device_tpe, sizeof(graph_node)*edge_counter, cudaMemcpyDeviceToHost);
		cudaMemcpy(device_tpe, tpe, sizeof(graph_node)*edge_counter, cudaMemcpyHostToDevice);
		cudaMemset(device_queueCounter, 0, sizeof(unsigned int));
		cudaMemcpy(&queueCounter, device_queueCounter, sizeof(uint), cudaMemcpyDeviceToHost);
	}

	cudaMemcpy(initDist, distance_cur, sizeof(uint)*graph->size(), cudaMemcpyDeviceToHost);

	for(int i=0; i < graph->size(); i++){
		if(initDist[i] == UINT_MAX){
		    outputFile << i << ":" << "INF" << endl;
		}
		else{
		    outputFile << i << ":" << initDist[i] << endl;
		}
	}
	//http://www.geeksforgeeks.org/g-fact-30/
	free(initDist);
	cudaFree(distance_cur);
	cudaFree(distance_prev);
	free(tpe);
	//free(nodeQueue);
	// free(allOffsets);
	// free(initDist);
	// cout<<"test";
	// cudaFree(distance_cur);
	// cudaFree(distance_prev);
	// cudaFree(device_tpe);
	// cudaFree(device_nodeQueue);
}
