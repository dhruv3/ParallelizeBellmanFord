#include <vector>
#include <iostream>

#include "utils.h"
#include "cuda_error_check.cuh"
#include "initial_graph.hpp"
#include "parse_graph.hpp"

__global__ void pulling_kernel(std::vector<initial_vertex> * peeps, int offset, int * anyChange){

    //update me based on my neighbors. Toggle anyChange as needed.
    //offset will tell you who I am.
}

void puller(std::vector<initial_vertex> * peeps, int blockSize, int blockNum){
    setTime();

    /*
     * Do all the things here!
     **/

    std::cout << "Took " << getTime() << "ms.\n";
}
