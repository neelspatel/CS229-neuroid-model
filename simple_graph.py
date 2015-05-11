import numpy as np
import time
import gen_graph

def sample_r_randomly(vertices, r):
    idx = np.random.choice(len(vertices), r)   
    #return vertices[idx]
    return [vertices[i] for i in idx]       

#counts how many vertices in this item have enough inputs from vertices in the other item.
#'Enough inputs from' means that the sum of edge weights is greater than threshold/max synapse strength
def getPotentialOverlap(fromItem, toItem, incoming_vertices, outgoing_vertices, items, maxSynapseStrength=1.0, threshold=5.0):
    neurons_from = items[fromItem]
    neurons_to = items[toItem]

    number_connected = 0.0
    total_number = 0

    for neuron in neurons_to:
        #first checks if this neuron is in both items
        if neuron in neurons_from:
            number_connected += 1
        else:
            #gets the list of connections to this neuron in the toItem
            connections = incoming_vertices[neuron]

            #gets the list of connections to this neuron from a neuron in fromItem
            shared_connections = [x for x in connections if x in neurons_from]               

            #if the potential weight is large enough to activate the threshold
            if len(shared_connections) >= threshold/maxSynapseStrength:
                number_connected += 1

            total_number += len(shared_connections)

    try:
        return number_connected / len(neurons_to)
    except ZeroDivisionError:
        print "Error"
        return 0



def testPotentialOverlap(graph=None, l = 100, w = 100, degree = 100, r = 200, n = 1):
    start = time.time()
    
    if graph:
        incoming_vertices, outgoing_vertices = graph
    else:
        incoming_vertices, outgoing_vertices = gen_graph.gen_graph_vertices(l,w,degree)
    
    elapsed = time.time() - start
    print "%fs (l=%d, w=%d, degree=%d, r=%d)" % (elapsed, l, w, degree, r)


    points = np.arange(l*w)
    overlaps = [0 for i in range(n)]

    for i in range(n):
        #generates two items
        items = {
            1: set(sample_r_randomly(points, r)),
            2: set(sample_r_randomly(points, r))
        }

        overlaps[i]  = getPotentialOverlap(1, 2, incoming_vertices, outgoing_vertices, items)

    return overlaps

#overlaps = testPotentialOverlap(n=10)

#from the set of vertices, what items are active
def getActiveItems(vertices, items, threshold=0.8):
    vertices_set = set(vertices)
    active_items = []

    for item in items.keys():
        item_vertices = items[item]

        overlap = vertices_set.intersect(item_vertices)

        #if more than threshold percentage of the neurons in this item
        #are firing
        if len(overlap)/len(item_vertices) > threshold:
            active_items.append(item)

    return active_items

#at every time point of the simulaiton, what are the items that are supposed to be active

#to run simulation
#create some amount of items from r random vertices, keyed by item id
#create amount of random associations (from one item id to another item id)
#function that creates association: turn on this item, keep dictionary of items
#and items that they are associated with
def simulation(graph=None, l = 100, w = 100, degree = 100, r = 200, num_items=2, num_associations=2):
    start = time.time()
    
    if graph:
        incoming_vertices, outgoing_vertices = graph
    else:
        incoming_vertices, outgoing_vertices = gen_graph.gen_graph_vertices(l,w,degree)
    
    elapsed = time.time() - start
    print "%fs (l=%d, w=%d, degree=%d, r=%d)" % (elapsed, l, w, degree, r)


    points = np.arange(l*w)
    overlaps = [0 for i in range(n)]

    items = {}

    #holds associations for key fromItem to a list of toItems
    associations = {}

    #creates random items
    for i in range(num_items):
        items[i] = set(sample_r_randomly(points, r))

    #creates random associations between items
    for i in range(num_associations):
        fromItem, toItem = np.random.choice(num_items, 2)

        if fromItem not in associations:
            associations[fromItem] = []

        associations[fromItem].append(toItem)

    

       



