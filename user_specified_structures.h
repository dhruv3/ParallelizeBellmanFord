#ifndef	USER_SPECIFIED_STRUCTURES_H
#define	USER_SPECIFIED_STRUCTURES_H

/**************************************
 *  STRUCTURES
 **************************************/


// Vertex structure.
struct Vertex{

	unsigned int distance;
};

// Vertex_static structure. Those properties of the vertex that remain constant during processing should be declared here.
typedef struct Vertex_static{

}Vertex_static;

// Edge structure.
struct Edge{

	unsigned int weight;

};



#endif	//	USER_SPECIFIED_STRUCTURES_H
