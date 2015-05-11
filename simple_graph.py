import numpy as np
import time
import gen_graph
import random

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

        overlap = vertices_set.intersection(item_vertices)  

        #if more than threshold percentage of the vertices in this item
        #are firing
        if len(overlap)/float(len(item_vertices)) > threshold:
            active_items.append(item)

    return active_items

#at every time point of the simulaiton, what are the items that are supposed to be active

#to run simulation
#create some amount of items from r random vertices, keyed by item id
#create amount of random associations (from one item id to another item id)
#function that creates association: turn on this item, keep dictionary of items
#and items that they are associated with
def simulation(graph=None, l = 100, w = 100, degree = 100, r = 200, fire_thresh=0.85, num_items=2, num_associations=2, timesteps=50, max_weight=10):
    start = time.time()
    
    if graph:
        incoming_vertices, outgoing_vertices = graph
    else:
        incoming_vertices, outgoing_vertices = gen_graph.gen_graph_vertices(l,w,degree)
    
    weights = {}

    for coordinate in outgoing_vertices:
        for connection in outgoing_vertices[coordinate]:
            weights[(coordinate, connection)] = 1
    
    points = np.arange(l*w)

    items = {}

    #holds associations for key fromItem to a list of toItems
    associations = {}
    #keys tuples of memorizated items, value is the memorized conjunction
    memorizations = {}

    #creates random items
    for i in range(num_items):
        items[i] = set(sample_r_randomly(points, r))

    #creates random associations between items
    for i in range(num_associations):
        fromItem, toItem = np.random.choice(num_items, 2)
        if fromItem not in associations:
            associations[fromItem] = []
        associations[fromItem].append(toItem)

        #maxes the edge weight for each vertex combination
        fromItemVertices = items[fromItem]
        toItemVertices = items[toItem]

        for fromItemVertex in fromItemVertices:
            for toItemVertex in toItemVertices:
                if (fromItemVertex, toItemVertex) in weights:
                    weights[(fromItemVertex, toItemVertex)] = max_weight


    elapsed = time.time() - start
    print "%fs (l=%d, w=%d, degree=%d, r=%d)" % (elapsed, l, w, degree, r)


    #runs the simulation for each time step

    #start with a random item, and fire its vertices    
    random_item = random.randint(0,num_items-1)
    firing_items = [random_item]
    firing_vertices = fire_item(items[random_item], fire_thresh)

    for i in range(timesteps):     
        #gets the list of associated items for each currently firing item
        associated_items = []
        for x in firing_items:
            associated_items += associations.get(x, [])
        associated_items = set(associated_items)

        #gets the next set of firing vertices
        firing_vertices = get_active_set((incoming_vertices, outgoing_vertices), firing_vertices, weights)

        #gets the list of items from the associated firing vertices
        firing_items = set(getActiveItems(firing_vertices, items))
        print len(firing_vertices), len(associated_items), len(firing_items.intersection(associated_items)), len(firing_items)

# how do we determine if we should be activating a memorization?
# alternatively, in each timestep we could simulate a few things, i.e, particular associations and memorizations

def simulate_memorization_on(graph, weights, item1, item2, mem_item, items, r, fire_thresh=0.85)
    """
    we assume that mem_item is stored as the meorization of item1 and item2
    will activate the nodes in item1 and item2, see if we activate nodes in mem_item
    only deals with ON error currently
    """
    # activate neurons of both item sets
    firing_vertices = set.union(fire_item(item1, fire_thresh), fire_item(item2, fire_thresh))
    firing_vertices = get_active_set((incoming_vertices, outgoing_vertices), firing_vertices, weights)
    firing_items = getActiveItems(firing_vertices)
    if mem_item in firing_items: 
        return True
    else:
        return False

def simulate_memorization_off(graph, weights, item1, item2, mem_item, items, r, fire_thresh=0.85):
    """ 
    simulate memorization with 
    """
    return None


def get_active_set(graph, firing_vertices, weights = False, threshold = 50):
    """
    runs the simulation for one time point, with the firing_vertices being the list/set of nodes that are active in the current time point
    threshold, min number of active neighbors
    weights: if false, defaults to 1
    returns the set of nodes that are firing at next time point
    """
    incoming_vertices, outgoing_vertices =  graph
    next_firing = set([])
    thresh = {}
    for vert in firing_vertices:
        for cur_vert in outgoing_vertices[vert]:
            if cur_vert not in next_firing:
                try: 
                    if weights:
                        thresh[cur_vert] += weights[(vert, cur_vert)]
                    else:
                        thresh[cur_vert] += 1
                except KeyError:
                    if weights:
                        thresh[cur_vert] = weights[(vert, cur_vert)]
                    else:
                        thresh[cur_vert] = 1
                if thresh[cur_vert] >= threshold:
                    next_firing.add(cur_vert)
    return next_firing

def fire_item(item, fire_thresh = 0.85):
    """
    returns a random set of fire_thresh nodes to fire for an item 
    """
    num_verts = int(len(item) * fire_thresh)
    return set(list(np.random.choice(np.array(list(item)), num_verts, replace=False)))

def test_memorization(graph, item1, item2, items, r):
    """
    returns list of nodes that can be activated by item1 and item 2
    if not possible, returns None
    """
    item1_actives = get_active_set(graph, fire_item(items[item1]), weights=False, threshold=5)
    item2_actives = get_active_set(graph, fire_item(items[item2]), weights=False, threshold=5)
    candidate_verts = intersection(item1_actives, item2_active)
    if len(candidate_verts) >= r:
        return candidate_verts 
    else:
        return None

def memorize(graph, item1, item2, items, r):
    """
    returns a set of vertices that can serve as the conjunction of items 1 and 2
    does not need to modify graph, since weights are max 
    """
    candidate_verts = test_memorization(graph, item1, item2, items, r)
    if candidate_verts:
        return set(list(candidate_verts[:r]))
    else:
        return None

def make_memorization(graph, weights, item1, item2, items, r):
    """
    Finds a set of r vertices connected to items1 and 2, creates a new item 
    """
    candidate_verts = memorize(graph, item1, item2, items, r)
    if candidate_verts:
        mem_item = candidate_verts
        items.append(mem_item)
        mem_item_index = len(items)-1 
        for item_vert in set.union(items[item1], items[item2]):
            for mem_vert in candidate_verts:
                if (item_vert, mem_vert) in weights:
                    weights[(item_vert, mem_vert)] = max_synapse_strength
    else:
        # memorization was not possible for these two items
        return None
    return graph, weights, mem_item_index, items



def test_association(graph, item1, item2, items):
    item1_actives = get_active_set(graph, items[item1], weights=False, threshold=5)
    if item2 in get_active_items(item1_actives, items):
        return True
    else:
        return False


def get_memorization_overlap(fromItem, toItem, incoming_vertices, outgoing_vertices, items, maxSynapseStrength=1.0, threshold=5.0):
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

def test_memorization_overlap(graph=None, l = 100, w = 100, degree = 100, r = 200, n = 1):
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
        
        # want to find the set of neurons that are sufficiently connected to item1 and item2

        overlapping_nodes = test_memorization(graph, 1, 2, items, r)
        if overlapping_nodes:
            overlaps[i]  = len(overlapping_nodes)
        else:
            overlaps[i] = 0
    return overlaps


