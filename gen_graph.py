from numpy.random import multinomial
import numpy as np
from graph import Graph
import cPickle as pickle
import time
import math

def euc_dist(p1, p2):
    """
    returns euclidean distance btween p1, p2
    """
    p1x, p1y = p1
    p2x, p2y = p2
    return (p1x-p2x)**2 + (p1y-p2y)**2

def mh_dist(p1, p2):
    """
    manhattan distance betweeen two grid points
    """
    return abs(p1x-p2x) + abs(p1y-p2y)

def unorm_prob(dist, graph):
    """
    what is the edge probability given this distance
    """
    #return np.ones(len(dist))
    return (25.0/(dist + 0.01))**0.5

    if graph:
        for y in range(l):
            for x in range(w):
                graph.add_vertex((l,w))   # initialize graph

#generates degrees of every multiple of 2 from 32 to size
def generate_degrees(size,upper_bound=10):
    max_degree_exponent = min(upper_bound, int(math.log(size*size, 2)))
    return [int(math.pow(2,i)) for i in range(5, max_degree_exponent+1)]

def generate_test_graphs(sizes=[150,200,250,300]):
    for size in sizes:
        degrees = generate_degrees(size)
        for degree in degrees:
            if not (degree==256 and size==300):

                start = time.time()
                incoming_vertices, outgoing_vertices = gen_graph_vertices(size, size, degree)
                elapsed = time.time() - start

                pickle.dump(incoming_vertices, open("graphs/" + str(size) + "_" + str(degree) + "_incoming_vertices.p", "wb"))
                pickle.dump(outgoing_vertices, open("graphs/" + str(size) + "_" + str(degree) + "_outgoing_vertices.p", "wb"))    

                print size, degree, elapsed

def read_test_graph(size, degree):
    incoming_vertices = pickle.load(open("graphs/" + str(size) + "_" + str(degree) + "_incoming_vertices.p", "rb"))
    outgoing_vertices = pickle.load(open("graphs/" + str(size) + "_" + str(degree) + "_outgoing_vertices.p", "rb"))    
    return incoming_vertices, outgoing_vertices


def gen_graph_vertices(l, w, degree):
    """
    generate a graph from grid using the probability distribution above 
    """    
    incoming_vertices = {}
    outgoing_vertices = {}

    num_points = l * w
    pts = np.arange(num_points)
    dist_table = np.zeros((2*l+1, 2*w+1))
    for y in range(2*l+1):
        for x in range(2*w+1):
            dist_table[x, y] = float((y-l)**2 + (x-w)**2)
    dist_table = unorm_prob(dist_table, None)

    for i in range(num_points):
        # add the appropriate vertices
        x, y = i % w, i / l
        dist_probs = dist_table[w-x:2*w-x, l-y:2*l-y].flatten()
        dist_probs /= dist_probs.sum()
        verts = dist_probs.cumsum().searchsorted(np.random.sample(degree))

        for v in verts:      
            if i not in outgoing_vertices:                
                outgoing_vertices[i] = {v: 1}
            else:
                outgoing_vertices[i][v] = 1

            if v not in incoming_vertices:
                incoming_vertices[v] = {i: 1}
            else:
                incoming_vertices[v][i] = 1

    return incoming_vertices, outgoing_vertices

def gen_random_graph(l, w, degree):
    """
    generate a random graph
    """
    graph = Graph()

    num_points = l * w
    pts = np.arange(num_points)    

    for i in range(num_points):
        # add the appropriate vertices
        x, y = i % w, i / l
        
        verts = np.random.choice(pts, degree)

        for v in verts:            
            graph.addEdge((v/w, v%w), (x, y))
    return graph

def gen_random_graph_vertices(l, w, degree):
    """
    generate a graph from grid using the probability distribution above 
    """    
    incoming_vertices = {}
    outgoing_vertices = {}

    num_points = l * w
    pts = np.arange(num_points)

    for i in range(num_points):
        # add the appropriate vertices
        x, y = i % w, i / l
        verts = np.random.choice(pts, degree)

        for v in verts:      
            if i not in outgoing_vertices:                
                outgoing_vertices[i] = {v: 1}
            else:
                outgoing_vertices[i][v] = 1

            if v not in incoming_vertices:
                incoming_vertices[v] = {i: 1}
            else:
                incoming_vertices[v][i] = 1

    return incoming_vertices, outgoing_vertices

#returns edges and vertices
def gen_graph_edges(l, w, degree):
    """
    generate a graph from grid using the probability distribution above 
    """    
    edges = []

    num_points = l * w
    pts = np.arange(num_points)
    dist_table = np.zeros((2*l+1, 2*w+1))
    for y in range(2*l+1):
        for x in range(2*w+1):
            dist_table[x, y] = float((y-l)**2 + (x-w)**2)
    dist_table = unorm_prob(dist_table, None)

    for i in range(num_points):
        # add the appropriate vertices
        x, y = i % w, i / l
        dist_probs = dist_table[w-x:2*w-x, l-y:2*l-y].flatten()
        dist_probs /= dist_probs.sum()
        verts = dist_probs.cumsum().searchsorted(np.random.sample(degree))

        for v in verts:            
            edges.append(((v/w, v%w), (x, y)))
    return edges

def gen_graph(l, w, degree):
    """
    generate a graph from grid using the probability distribution above 
    """
    graph = Graph()

    num_points = l * w
    pts = np.arange(num_points)
    dist_table = np.zeros((2*l+1, 2*w+1))
    for y in range(2*l+1):
        for x in range(2*w+1):
            dist_table[x, y] = float((y-l)**2 + (x-w)**2)
    dist_table = unorm_prob(dist_table, graph)

    for i in range(num_points):
        # add the appropriate vertices
        x, y = i % w, i / l
        dist_probs = dist_table[w-x:2*w-x, l-y:2*l-y].flatten()
        dist_probs /= dist_probs.sum()
        verts = dist_probs.cumsum().searchsorted(np.random.sample(degree))

        for v in verts:            
            graph.addEdge((v/w, v%w), (x, y))
    return graph

def gen_random_graph(l, w, degree):
    """
    generate a random graph
    """
    graph = Graph()

    num_points = l * w
    pts = np.arange(num_points)    

    for i in range(num_points):
        # add the appropriate vertices
        x, y = i % w, i / l
        
        verts = np.random.choice(pts, degree)

        for v in verts:            
            graph.addEdge((v/w, v%w), (x, y))
    return graph

#to save a graph, we only need to save edges
def pickle_graph_edges(edges, directory, filename):
    pickle.dump(edges, open(directory + "/" + filename, "wb"))

#loads a graph from pickled edges
def unpickle_graph_edges(directory, filename):
    start = time.time()
    edges = pickle.load(open(directory + "/" + filename, "rb"))
    print time.time() - start, "unpickling"

    graph = Graph()

    start = time.time()
    for from_vertex, to_vertex in edges:
        graph.addEdge(from_vertex, to_vertex)
    print time.time() - start, "loading"

    return graph

#generate a graph
"""
generated_graph = gen_graph.gen_graph(100,100,10)
graph.test_overlap(generated_graph, 2500, 1)
"""

#get list of vertices
#generate random items
#check the overlap of those items
