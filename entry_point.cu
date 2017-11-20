#include <cstring>
#include <stdexcept>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#include "utils.h"
#include "cuda_error_check.cuh"
#include "initial_graph.hpp"
#include "parse_graph.hpp"

#include "opt.cu"
#include "impl2.cu"
#include "impl1.cu"


enum class ProcessingType {Push, Neighbor, Own, Unknown};
enum SyncMode {InCore, OutOfCore};
enum SyncMode syncMethod;
enum SmemMode {UseSmem, UseNoSmem};
enum SmemMode smemMethod;

// Open files safely.
template <typename T_file>
void openFileToAccess( T_file& input_file, std::string file_name ) {
	input_file.open( file_name.c_str() );
	if( !input_file )
		throw std::runtime_error( "Failed to open specified file: " + file_name + "\n" );
}

void testCorrectness(graph_node *edges, const char* outputFileName, uint nVertices, uint nEdges) {
	
	std::cout << std::endl << "TESTING CORRECTNESS" << std::endl;
	std::cout << "RUNNING SEQUENTIAL BMF..." << std::endl;
	
	unsigned int *d= new unsigned int[nVertices];
	
	d[0]=0;
	for (int i = 1; i < nVertices; ++i){
		d[i] = UINT_MAX;
	}

	int change = 0;
	for(int i = 1; i < nVertices; i++){
		for(int j = 0; j < nEdges; j++){
			int u = edges[j].src;
			int v = edges[j].dst;
			int w = edges[j].weight;
			if(d[u] == UINT_MAX){
				continue;
			} else if(d[u]+w < d[v]){
				d[v] = d[u]+w;
				change = 1;
			}
		}
		if(!change){
			break;
		}
		change = 0;
	}
	
	//Compare the distance array and the parallel output file
	std::ifstream outputFile;
	openFileToAccess< std::ifstream >( outputFile, std::string( outputFileName ) );

	std::string line;
	int i = 0;
	int incorrect = 0;
	while (getline(outputFile,line)) {
		std::string curr = (d[i] < UINT_MAX) ? (std::to_string(i) + ":" + std::to_string(d[i])):(std::to_string(i) +":" + "INF");

		if(line.compare(curr) != 0) {
			incorrect++;
    		//std::cout << "Correct: " << curr << "\tYours: " << line << std::endl;
		}
		i++;
	}
	if(i != nVertices) {
		std::cout << "Insufficient vertices found in outputfile" << std::endl;
		std::cout << "Expected: " << nVertices << "Found: " << i << std::endl;
		return;
	}
	std::cout << "Correct: " << std::to_string(nVertices-incorrect) << "\t Incorrect: " << std::to_string(incorrect) << " \t Total: " << std::to_string(nVertices) << std::endl;
	outputFile.close();
	delete[] d;
}

void pull_edges(std::vector<initial_vertex>& graph, graph_node* edge_list, unsigned int edge_num){

	unsigned int k = 0;

	for(int i = 0 ; i < graph.size() ; i++){
		std::vector<neighbor> nbrs = (graph)[i].nbrs;
	    for(int j = 0 ; j < nbrs.size() ; j++, k++){
			edge_list[k].src = nbrs[j].srcIndex;
			edge_list[k].dst = i;
			edge_list[k].weight = nbrs[j].edgeValue.weight;
	    }
	}

	if( k != edge_num )
	    printf("ERROR: Edge numbers don't match up\n");
}

// Execution entry point.
int main( int argc, char** argv )
{

	std::string usage =
		"\tRequired command line arguments:\n\
			Input file: E.g., --input in.txt\n\
                        Block size: E.g., --bsize 512\n\
                        Block count: E.g., --bcount 192\n\
                        Output path: E.g., --output output.txt\n\
			Processing method: E.g., --method bmf (bellman-ford), or tpe (to-process-edge), or opt (one further optimizations)\n\
			Shared memory usage: E.g., --usesmem yes, or no \n\
			Sync method: E.g., --sync incore, or outcore\n";

	try {

		std::ifstream inputFile;
		std::ofstream outputFile;
		std::string outputFileName;
		int selectedDevice = 0;
		int bsize = 0, bcount = 0;
		int vwsize = 32;
		int threads = 1;
		long long arbparam = 0;
		bool nonDirectedGraph = false;		// By default, the graph is directed.
		ProcessingType processingMethod = ProcessingType::Unknown;
		syncMethod = OutOfCore;


		/********************************
		 * GETTING INPUT PARAMETERS.
		 ********************************/

		for( int iii = 1; iii < argc; ++iii )
			if ( !strcmp(argv[iii], "--method") && iii != argc-1 ) {
				if ( !strcmp(argv[iii+1], "bmf") )
				        processingMethod = ProcessingType::Push;
				else if ( !strcmp(argv[iii+1], "tpe") )
    				        processingMethod = ProcessingType::Neighbor;
				else if ( !strcmp(argv[iii+1], "opt") )
				    processingMethod = ProcessingType::Own;
				else{
           std::cerr << "\n Un-recognized method parameter value \n\n";
           exit;
         }   
			}
			else if ( !strcmp(argv[iii], "--sync") && iii != argc-1 ) {
				if ( !strcmp(argv[iii+1], "incore") )
				        syncMethod = InCore;				
				if ( !strcmp(argv[iii+1], "outcore") )
    				        syncMethod = OutOfCore;				
				else{
           std::cerr << "\n Un-recognized sync parameter value \n\n";
           exit;
         }  

			}
			else if ( !strcmp(argv[iii], "--usesmem") && iii != argc-1 ) {
				if ( !strcmp(argv[iii+1], "yes") )
				        smemMethod = UseSmem;
				if ( !strcmp(argv[iii+1], "no") )
    				        smemMethod = UseNoSmem;
        else{
           std::cerr << "\n Un-recognized usesmem parameter value \n\n";
           exit;
         }  
			}
			else if( !strcmp( argv[iii], "--input" ) && iii != argc-1 /*is not the last one*/)
				openFileToAccess< std::ifstream >( inputFile, std::string( argv[iii+1] ) );
			else if( !strcmp( argv[iii], "--output" ) && iii != argc-1 /*is not the last one*/){
				openFileToAccess< std::ofstream >( outputFile, std::string( argv[iii+1] ) );
				outputFileName = std::string(argv[iii+1]);
			}
			else if( !strcmp( argv[iii], "--bsize" ) && iii != argc-1 /*is not the last one*/)
				bsize = std::atoi( argv[iii+1] );
			else if( !strcmp( argv[iii], "--bcount" ) && iii != argc-1 /*is not the last one*/)
				bcount = std::atoi( argv[iii+1] );

		if(bsize <= 0 || bcount <= 0){
			std::cerr << "Usage: " << usage;
      exit;
			throw std::runtime_error("\nAn initialization error happened.\nExiting.");
		}
		if( !inputFile.is_open() || processingMethod == ProcessingType::Unknown ) {
			std::cerr << "Usage: " << usage;
			throw std::runtime_error( "\nAn initialization error happened.\nExiting." );
		}
		if( !outputFile.is_open() )
			openFileToAccess< std::ofstream >( outputFile, "out.txt" );
		( cudaSetDevice( selectedDevice ) );
		std::cout << "Device with ID " << selectedDevice << " is selected to process the graph.\n";


		/********************************
		 * Read the input graph file.
		 ********************************/

		std::cout << "Collecting the input graph ...\n";
		std::vector<initial_vertex> parsedGraph( 0 );
		uint nEdges = parse_graph::parse(
				inputFile,		// Input file.
				parsedGraph,	// The parsed graph.
				arbparam,
				nonDirectedGraph );		// Arbitrary user-provided parameter.
		std::cout << "Input graph collected with " << parsedGraph.size() << " vertices and " << nEdges << " edges.\n";


		/********************************
		 * Process the graph.
		 ********************************/


		switch(processingMethod){
			case ProcessingType::Push:
				if(smemMethod == UseSmem){
					cout << "USE SMEM" << endl;
					puller_smem(&parsedGraph, bsize, bcount, outputFile);
				} 
				else if(syncMethod == OutOfCore){
					puller(&parsedGraph, bsize, bcount, outputFile);
				} 
				else if(syncMethod == InCore){
					puller_incore(&parsedGraph, bsize, bcount, outputFile);
				}
				else {
					cout << "syncMethod not specified" << endl;
					exit(EXIT_FAILURE);
				}
			    break;
			case ProcessingType::Neighbor:
				if(syncMethod == OutOfCore){
					puller_outcore_impl2(&parsedGraph, bsize, bcount, outputFile);
				} 
				else if(syncMethod == InCore){
					puller_incore_impl2(&parsedGraph, bsize, bcount, outputFile);
				}
				else {
					cout << "syncMethod not specified" << endl;
					exit(EXIT_FAILURE);
				}
			    break;
			default:
			    own(&parsedGraph, bsize, bcount);
		}

		/********************************
		 * It's done here.
		 ********************************/
		graph_node *testEdgeList = new graph_node[nEdges];
		pull_edges(parsedGraph, testEdgeList, nEdges);
		testCorrectness(testEdgeList, outputFileName.c_str(), parsedGraph.size(), nEdges);
		CUDAErrorCheck( cudaDeviceReset() );
		std::cout << "Done.\n";
		return( EXIT_SUCCESS );

	}
	catch( const std::exception& strException ) {
		std::cerr << strException.what() << "\n";
		return( EXIT_FAILURE );
	}
	catch(...) {
		std::cerr << "An exception has occurred." << std::endl;
		return( EXIT_FAILURE );
	}

}
