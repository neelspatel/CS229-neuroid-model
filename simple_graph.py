import numpy as np
import time
import gen_graph
import random
import copy 

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

#to test the simulation:
# - generate a graph, and a random two items, which we will associate
# - for each trial, turn on 80% of the first item randomly, check if the second item is on

def simulation(graph=None, l = 100, w = 100, degree = 100, r = 200, fire_thresh=0.85, num_items=2, num_associations=2, num_iterations=50, num_checks= 20, max_weight=10, threshold=20):
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

    #holds associations for a list of (fromItem, toItem) tuples
    associationsList = []

    #for a key fromItem, holds a list of toItems to which it is associated
    associationsDict = {}
    
    #keys tuples of memorizated items, value is the memorized conjunction
    memorizations = {}

    #creates random items
    for i in range(num_items):
        items[i] = set(sample_r_randomly(points, r))

    #creates random associations between items
    num_failed_associations = 0.0
    num_failed_memorizations = 0.0

    for i in range(num_memorizations):
        item1, item2 = np.random.choice(num_items, 2, replace=False)  
        graph, weights, mem_item_index, items = make_memorization(graph, weights, item1, item2, items, r)
        # keep picking until we get items that can be memorized
        while mem_item_index is None:
            num_failed_memorizations += 1
            item1, item2 = np.random.choice(num_items, 2, replace=False)  
            graph, weights, mem_item_index, items = make_memorization(graph, weights, item1, item2, items, r)
        memorizations[(item1, item2)] = mem_item_index

    for i in range(num_associations):
        fromItem, toItem = np.random.choice(num_items, 2, replace=False)  

        #keep choosing items until we find items that can be associated with each other
        potential_overlap = getPotentialOverlap(fromItem, toItem, incoming_vertices, outgoing_vertices, items, maxSynapseStrength=max_weight, threshold=threshold)
        
        while potential_overlap < 0.8:
            num_failed_associations += 1

            fromItem, toItem = np.random.choice(num_items, 2, replace=False)  

            #keep choosing items until we find items that can be associated with each other
            potential_overlap = getPotentialOverlap(fromItem, toItem, incoming_vertices, outgoing_vertices, items, maxSynapseStrength=max_weight, threshold=threshold)
            

        associationsList.append((fromItem, toItem))

        if fromItem not in associationsDict:
            associationsDict[fromItem] = []

        associationsDict[fromItem].append(toItem)

        #maxes the edge weight for each vertex combination
        fromItemVertices = items[fromItem]
        toItemVertices = items[toItem]

        for fromItemVertex in fromItemVertices:
            for toItemVertex in toItemVertices:
                if (fromItemVertex, toItemVertex) in weights:
                    weights[(fromItemVertex, toItemVertex)] = max_weight

    elapsed = time.time() - start
    print "%fs (l=%d, w=%d, degree=%d, r=%d)" % (elapsed, l, w, degree, r)

    print num_failed_associations, "failed associations;", num_associations, "successful associations"


    #runs the simulation
    num_successful_on = 0.0
    num_successful_off = 0.0

    for i in range(num_iterations):

        #choose a random association   
        random_association_index = random.randint(0,num_associations-1)        
        fromItem, toItem = associationsList[random_association_index]

        num_successful_on += simulate_association_on(graph, weights, fromItem, toItem, items, fire_thresh, num_checks=num_checks)
        num_successful_off += simulate_association_off(graph, weights, fromItem, toItem, items, associationsDict, fire_thresh, num_checks=num_checks)        
        
        # choose a memorization
        item1, item2 = random.choice(memorization.keys())
        mem_item = memorizations[(item1, item2)]
        mem_success_on = simulate_memorization_on(graph, weights, item1, item2, mem_item, items, r, trials=num_checks)
        mem_success_off = simulate_memorization_off(graph, weights, item1, item2, mem_item, items, r, trials=num_checks)
                
    return num_failed_associations/(num_failed_associations+num_associations), num_successful_on/(num_iterations * num_checks), num_successful_off/(num_iterations * num_checks)



#returns True of False based on if there is a simulated association between fromItem and toItem
def simulate_association_on(graph, weights, fromItem, toItem, items, fire_thresh=0.85, threshold=20, num_checks=20):
    incoming_vertices, outgoing_vertices = graph

    #gets the vertices for each item
    fromItemVertices = items[fromItem]

    num_successful_checks = 0.0
    overlap_amount = 0.0
    overlaps = []

    #fires the set of firing vertices
    for i in range(num_checks):
        firing_vertices = fire_item(fromItemVertices, fire_thresh)

        #gets the next set of firing vertices
        firing_vertices = get_active_set((incoming_vertices, outgoing_vertices), firing_vertices, weights, threshold)

        #gets the list of items from the associated firing vertices
        firing_items = set(getActiveItems(firing_vertices, items))

        num_successful_checks += (toItem in firing_items)

        #calculates the amount of vertices firing
        correctly_firing = len(firing_vertices.intersection(items[toItem])) / float(len(items[toItem]))
        overlap_amount += correctly_firing

        overlaps.append(correctly_firing)

    #checks if item 2 was identified as associated
    #return num_successful_checks
    return overlap_amount

#returns True of False based on if there is not a simulated association between fromItem and toItem
def simulate_association_off(graph, weights, fromItem, toItem, items, associationsDict, fire_thresh=0.85, threshold=20, num_other_items=0, num_checks=20):
    incoming_vertices, outgoing_vertices = graph

    #gets the vertices for each item
    fromItemVertices = items[fromItem]    

    #fire num_other_items other random items, as long as these items
    #are not associated with toItem
    item_keys = items.keys()
    item_keys.remove(fromItem)
    item_keys.remove(toItem)

    num_successful_checks = 0.0
    overlap_amount = 0.0
    overlaps = []

    for j in range(num_checks):
        #fires only a small portion of the set of firing vertices
        firing_vertices = fire_item(fromItemVertices, 1-fire_thresh)


        for i in range(num_other_items):
            random_item_index = random.randint(0, len(item_keys)-1)
            random_item = item_keys[random_item_index]

            #keeps choosing a random item until we find one that is not 
            #associated with toItem
            while random_item in associationsDict and toItem in associationsDict[random_item]:
                random_item_index = random.randint(0, len(item_keys))
                random_item = item_keys[random_item_index]
                random_item_vertices = items[random_item]

            #turns this random item on
            firing_vertices = set.union(firing_vertices, fire_item(random_item_vertices, fire_thresh))


        #gets the next set of firing vertices
        firing_vertices = get_active_set((incoming_vertices, outgoing_vertices), firing_vertices, weights, threshold)

        #gets the list of items from the associated firing vertices
        firing_items = set(getActiveItems(firing_vertices, items))

        #checks if item 2 was not identified as associated
        num_successful_checks += (toItem in firing_items)

        #calculates the amount of vertices firing
        correctly_firing = len(firing_vertices.intersection(items[toItem])) / float(len(items[toItem]))
        overlap_amount += correctly_firing

    #return num_successful_checks
    return overlap_amount


def simulate_one(graph=None, l = 100, w = 100, degree = 100, r = 200, fire_thresh=0.85, max_weight=10):
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

    #creates two random items
    item1 = set(sample_r_randomly(points, r))
    item2 = set(sample_r_randomly(points, r))

    items = {1: item1, 2: item2}
    

    #creates an associations between these two items
    #associations[item1].append(item2)   

    for fromItemVertex in item1:
        for toItemVertex in item2:
            if (fromItemVertex, toItemVertex) in weights:
                weights[(fromItemVertex, toItemVertex)] = max_weight


    elapsed = time.time() - start
    print "%fs (l=%d, w=%d, degree=%d, r=%d)" % (elapsed, l, w, degree, r)


    #runs the simulation for each time step

    #start with a random item, and fire its vertices        
    firing_vertices = fire_item(item1, fire_thresh)
    
    #gets the items associated with item1
    #associated_items = associations[item1]

    #gets the next set of firing vertices
    firing_vertices = get_active_set((incoming_vertices, outgoing_vertices), firing_vertices, weights)

    #gets the list of items from the associated firing vertices
    firing_items = set(getActiveItems(firing_vertices, items))

    print len(firing_vertices), "neurons firing"
    #print associatedItems, " (associated items)"
    print firing_items, " (firing items)"


def simulation_with_memorizations(graph=None, l = 100, w = 100, degree = 100, r = 200, fire_thresh=0.85, num_items=2, num_associations=2, num_memorizations=2, timesteps=50, max_weight=10):
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

    for i in range(num_memorizations):
        fromItem, toItem = np.random.choice(num_items, 2)
        if (fromItem, toItem) not in memorizations:
            # can they be memroized?
            graph, weights, mem_item, items = make_memorization(graph, weights, fromItem, toItem, items, r) 
            if mem_item:
                memorizations[(fromItem, toItem)] = mem_item
                # otherwise we made an error

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


def simulate_memorization_on(graph, weights, item1, item2, mem_item, items, r, fire_thresh=0.85, trials=1):
    """
    we assume that mem_item is stored as the meorization of item1 and item2
    will activate the nodes in item1 and item2, see if we activate nodes in mem_item
    only deals with ON error currently
    """
    results = []
    for trial in range(trials):
        incoming_vertices, outgoing_vertices = graph
        # activate neurons of both item sets
        firing_vertices = set.union(fire_item(items[item1], fire_thresh), fire_item(items[item2], fire_thresh))
        firing_vertices = get_active_set(graph, firing_vertices, weights)
        firing_rate = len(set.intersect(items[mem_item], firing_vertices)) / float(len(items[mem_item]))
        results.append(firing_rate)
    return results

def simulate_memorization_off(graph, weights, item1, item2, mem_item, items, r, fire_thresh=0.85, trials=1):
    """
    we assume that mem_item is stored as the meorization of item1 and item2
    will activate the nodes in item1 or item2 (or neither), want to ensure that mem_item does not fire
    """
    results = []
    no_fire_thresh = 1- fire_thresh
    for trial in range(trials):
        cur_result = [0.0, 0.0, 0.0]
        incoming_vertices, outgoing_vertices = graph
        # activate neurons of both item sets
        # first, turn off both vertex sets
        firing_vertices = set.union(fire_item(items[item1], no_fire_thresh), fire_item(items[item2], no_fire_thresh))
        firing_vertices = get_active_set(graph, firing_vertices, weights)
        cur_result[0] = len(set.intersect(items[mem_item], firing_vertices)) / float(len(items[mem_item]))
        # now, turn on one of the vertex sets 
        firing_vertices = set.union(fire_item(items[item1], no_fire_thresh), fire_item(items[item2], fire_thresh))
        firing_vertices = get_active_set(graph, firing_vertices, weights)
        cur_result[1] = len(set.intersect(items[mem_item], firing_vertices)) / float(len(items[mem_item]))
        # turn off the other vertex set 
        firing_vertices = set.union(fire_item(items[item1], fire_thresh), fire_item(items[item2], no_fire_thresh))
        firing_vertices = get_active_set(graph, firing_vertices, weights)
        cur_result[2] = len(set.intersect(items[mem_item], firing_vertices)) / float(len(items[mem_item]))
        results.append(cur_result)
    return results

def simulate_memorization_off_aux(graph, weights, item1, item2, mem_item, items, r, fire_thresh=0.85, trials=1):
    """
    we assume that mem_item is stored as the meorization of item1 and item2
    will deactive both items, activate random item that is not associated with mem_item
    """
    results = []
    no_fire_thresh = 1-fire_thresh
    for trial in range(trials):
        # choose random number of items to turn on
        aux_items = np.random.choice(items.keys(), random.choice(range(1, 4)))
        # check if these auxillary items are associated with mem_item 
        
        # activate neurons of both item sets
        # first, turn off both vertex sets
        firing_vertices = set.union(fire_item(items[item1], no_fire_thresh), fire_item(items[item2], no_fire_thresh))
        for aux_item in aux_items:
            firing_vertices = set.union(firing_vertices, fire_items(items[aux_item], fire_thresh))
        firing_vertices = get_active_set(graph, firing_vertices, weights)
        firing_items = getActiveItems(firing_vertices, items)
        if mem_item not in firing_items: 
            results.append(True)
        else:
            results.append(False)
    return results

def get_active_set(graph, firing_vertices, weights = False, threshold = 20):
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
    candidate_verts = set.intersection(item1_actives, item2_actives)
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
        return set(list(candidate_verts)[:r])
    else:
        return None

def make_memorization(graph, weights, item1, item2, items, r, max_synapse_strength=10):
    """
    Finds a set of r vertices connected to items1 and 2, creates a new item 
    """
    candidate_verts = memorize(graph, item1, item2, items, r)
    if candidate_verts:
        mem_item = candidate_verts
        mem_item_index = max(items.keys()) + 1
        items[mem_item_index] = mem_item
        start = time.time()
        for item_vert in set.union(items[item1], items[item2]):
            for mem_vert in candidate_verts:
                if (item_vert, mem_vert) in weights:
                    weights[(item_vert, mem_vert)] = max_synapse_strength/2
    else:
        # memorization was not possible for these two items
        return graph, weights, None, items
    return graph, weights, mem_item_index, items

def test_memorization_overlap(graph=None, l = 100, w = 100, degree = 100, r = 200, n = 1):
    """
    given a graph, picks n random items and tries to form their memorizations
    records if a memorization is possible 
    """
    if graph:
        incoming_vertices, outgoing_vertices = graph
    else:
        incoming_vertices, outgoing_vertices = gen_graph.gen_graph_vertices(l,w,degree)
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
            overlaps[i]  = len(overlapping_nodes) >= r
        else:
            overlaps[i] = False
    return overlaps

def test_association_noise(graph, weights, items, associations, num_aux_items, r):
    """
    want to turn on num_aux_items irrelevent items, for every item that should not be turned on, is it not actually turned on?
    """
    aux_items = np.random.choice(items.keys(), num_aux_items)
    unassociated_items = set(copy.copy(items.keys()))
    firing_vertices = set([])
    for aux_item in aux_items:
        firing_vertices = set.union(firing_vertices, fire_item(items[aux_item]))
        # want to find the items that are not in associations[item] for any item 
        for item in associations[aux_item]:
            unassociated_items.discard(item)
    overlap = []
    # what percentage of the neurons in these items are turned on
    new_firing_vertices = get_active_set(graph, firing_vertices, weights)
    for unassoc_item in unassociated_items:
        overlap.append(len(set.intersect(new_firing_vertices, items[unassoc_item])))
    return overlap
