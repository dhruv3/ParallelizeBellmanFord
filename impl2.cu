#include <vector>
#include <iostream>

#include "utils.h"
#include "cuda_error_check.cuh"
#include "initial_graph.hpp"
#include "parse_graph.hpp"

void puller_incore_impl2(std::vector<initial_vertex> * graph, int blockSize, int blockNum, ofstream &outputFile){
	
}

void puller_outcore_impl2(std::vector<initial_vertex> * graph, int blockSize, int blockNum, ofstream &outputFile){
	double t_filter, t_comp;
	t_filter = 0;
	t_comp = 0;

	unsigned int *initDist, *distance_cur, *distance_prev, *to_process_arr, *pred; 
	int *anyChange;
	int *checkIfChange = (int*)malloc(sizeof(int));
	graph_node *edge_list, *L, *T;
	unsigned int edge_counter, to_process_num;
	edge_counter = total_edges(*graph);
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
	qsort(edge_list, edge_counter, sizeof(graph_node), cmp_edge);			

	unsigned int *swapDistVariable = new unsigned int[graph->size()];
	unsigned int *hostTPA = new unsigned int[warp_num];

	cudaMalloc((void**)&to_process_arr, (size_t)sizeof(unsigned int) * warp_num);
	cudaMalloc((void**)&pred, (size_t)sizeof(unsigned int) * (graph->size()));
	cudaMalloc((void**)&L, (size_t)sizeof(graph_node)*edge_counter);
	cudaMalloc((void**)&distance_cur, (size_t)sizeof(unsigned int)*(graph->size()));
	cudaMalloc((void**)&distance_prev, (size_t)sizeof(unsigned int)*(graph->size()));
	cudaMalloc((void**)&anyChange, (size_t)sizeof(int));
	

	cudaMemcpy(distance_cur, initDist, (size_t)sizeof(unsigned int)*(graph->size()), cudaMemcpyHostToDevice);
	cudaMemcpy(distance_prev, initDist, (size_t)sizeof(unsigned int)*(graph->size()), cudaMemcpyHostToDevice);
	cudaMemcpy(L, edge_list, (size_t)sizeof(graph_node)*edge_counter, cudaMemcpyHostToDevice);
	cudaMemset(to_process_arr, 0, (size_t)sizeof(unsigned int)*warp_num);
	cudaMemset(pred, 0, (size_t)sizeof(unsigned int)*(graph->size()));
	cudaMemset(anyChange, 0, (size_t)sizeof(int));

	for(int i=0; i < ((int) graph->size())-1; i++){
		setTime();
		if(i == 0){
		    edge_process<<<blockNum,blockSize>>>(L, edge_counter, distance_prev, distance_cur, anyChange, pred);
		    cudaDeviceSynchronize();
		} 
		else {
		    cudaMemset(pred, 0, (size_t)sizeof(unsigned int)*(graph->size()));
		    edge_process<<<blockNum,blockSize>>>(L, edge_counter, distance_prev, distance_cur, anyChange, pred);
		    cudaDeviceSynchronize();
		    cudaFree(T);
		}
		t_comp += getTime();
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
		    cudaMemset(to_process_arr, 0, (size_t)sizeof(unsigned int)*warp_num);
		    set_wrap_count<<<blockNum, blockSize>>>(L, edge_counter, pred, to_process_arr);
		    cudaDeviceSynchronize();
		    cudaMemcpy(temp, to_process_arr + warp_num - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
		    to_process_num = *temp;
		    cudaMemcpy(hostTPA, to_process_arr, sizeof(unsigned int)*warp_num, cudaMemcpyDeviceToHost);
		    thrust::exclusive_scan(hostTPA, hostTPA + warp_num, hostTPA);
		    cudaMemcpy(to_process_arr, hostTPA, sizeof(unsigned int)*warp_num, cudaMemcpyHostToDevice);
		    cudaMemcpy(temp, to_process_arr + warp_num - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
		    to_process_num += *temp;
		    cudaMalloc((void**)&T, (size_t)sizeof(edge_node)*to_process_num);
		    filter_T<<<blockNum, blockSize>>>(L, edge_counter, pred, to_process_arr, T);
		    cudaDeviceSynchronize();
		    t_filter += getTime();
		}
	}

	printf("Computation Time: %f ms\nFiltering Time: %f ms\n", t_comp, t_filter);

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

	delete[] hostTPA;
	delete[] swapDistVariable;
	free(initDist);
	free(edge_list);
}