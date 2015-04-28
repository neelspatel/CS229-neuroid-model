import numpy as np
import time


class Vertex:
    def __init__(self,key):
        self.id = key
        self.outgoingEdges = {}
        self.incomingEdges = {}

    def addOutgoingNeighbor(self,nbr,weight=0):
        if nbr not in self.outgoingEdges:
            self.outgoingEdges[nbr] = 0            

        self.outgoingEdges[nbr] += weight   

    def addIncomingNeighbor(self,nbr,weight=0):
        if nbr not in self.incomingEdges:
            self.incomingEdges[nbr] = 0            

        self.incomingEdges[nbr] += weight      

    def __str__(self):
        return str(self.id) + ' outgoing: ' + str([x.id for x in self.outgoingEdges]) + ' incoming: ' + str([x.id for x in self.incomingEdges])

    def getOutgoingEdges(self):
        return self.outgoingEdges.keys()

    def getIncomingEdges(self):
        return self.incomingEdges.keys()

    def getId(self):
        return self.id

    def getOutgoingWeight(self,nbr):
        return self.outgoingEdges[nbr]

    def getIncomingWeight(self,nbr):
        return self.incomingEdges[nbr]

    def changeOutgoingWeightBy(self,nbr,weight, maxSynapseStrength):
        self.outgoingEdges[nbr] += weight
        self.outgoingEdges[nbr] = min(self.outgoingEdges[nbr], maxSynapseStrength)

    def changeIncomingWeightBy(self,nbr,weight, maxSynapseStrength):
        self.incomingEdges[nbr] += weight
        self.incomingEdges[nbr] = min(self.incomingEdges[nbr], maxSynapseStrength)

#an item is made of multiple vertices
class Item:
    def __init__(self,key):
        self.id = key
        self.vertList = set([])
        self.numVertices = 0

    def getId(self):
        return self.id

    #saves the vertex in the given vertex list
    def associateVertex(self,vertex):
        self.vertList.add(vertex)
        self.numVertices = self.numVertices + 1

    #removes the vertex from the vertex list
    def dissociateVertex(self,vertex):
        self.vertList.remove(vertex)
        self.numVertices = self.numVertices - 1   

    def getVertices(self):
        return self.vertList

    def getIncomingOverlap(self, comparisonItem, maxSynapseStrength=1.0, threshold=5.0):
        comparisonVertices = comparisonItem.getVertices()
        currentVertices = self.getVertices()

        number_connected = 0.0

        for i,vertex in enumerate(currentVertices):
            #first checks if this vertex is in both items
            if vertex in comparisonVertices:
                number_connected += 1
            else:
                #gets the list of connections shared between this vertex and the comparison item
                connections = vertex.getIncomingEdges()
                shared_connections = [x for x in connections if x in comparisonVertices]

                #sums the edge weights from each vertex in shared_connections to this vertex
                incoming_weight = sum([vertex.getIncomingWeight(x) for x in shared_connections])

                #if the potential weight is large enough to activate the threshold
                if incoming_weight >= threshold:
                    number_connected += 1

        return number_connected / len(currentVertices)

    #counts how many vertices in this item have enough inputs from vertices in the other item.
    #'Enough inputs from' means that the sum of edge weights is greater than threshold/max synapse strength
    def getPotentialOverlap(self, comparisonItem, maxSynapseStrength=1.0, threshold=5.0):
        comparisonVertices = comparisonItem.getVertices()
        currentVertices = self.getVertices()

        number_connected = 0.0

        for i,vertex in enumerate(currentVertices):
            #first checks if this vertex is in both items
            if vertex in comparisonVertices:
                number_connected += 1
            else:
                #gets the list of connections shared between this vertex and the comparison item
                connections = vertex.getIncomingEdges()
                shared_connections = [x for x in connections if x in comparisonVertices]               

                #if the potential weight is large enough to activate the threshold
                if len(shared_connections) >= threshold/maxSynapseStrength:
                    number_connected += 1

        return number_connected / len(currentVertices)
    
    def associateWith(self, comparisonItem, maxSynapseStrength=1.0, threshold=5.0):
        #for each edge from this to comparisonItem
        currentVertices = self.getVertices()
        comparisonVertices = comparisonItem.getVertices()


        for vertex in comparisonVertices:        
            #gets the list of connections from this item to the comparison item
            connections = vertex.getIncomingEdges()
            shared_connections = [x for x in connections if x in currentVertices]

            #sums the edge weights from each vertex in shared_connections to this vertex
            weight_into_comparison = sum([vertex.getIncomingWeight(x) for x in shared_connections])        

            #if we haven't hit the threshold yet, increase all edges so that we do
            #bounded by maxSynapseStrength
            if weight_into_comparison < threshold:
                try:
                    increase_amount = (threshold - weight_into_comparison) / len(shared_connections)
                except ZeroDivisionError:
                    increase_amount = 0

                for from_vertex in shared_connections:
                    from_vertex.changeOutgoingWeightBy(vertex, increase_amount, maxSynapseStrength)
                    vertex.changeIncomingWeightBy(from_vertex, increase_amount, maxSynapseStrength)

class Graph:
    def __init__(self):
        self.vertList = {}
        self.itemList = {}
        self.numVertices = 0
        self.numItems = 0

    def addVertex(self,key):
        self.numVertices = self.numVertices + 1
        newVertex = Vertex(key)
        self.vertList[key] = newVertex
        return newVertex

    def addItem(self,key):
        self.numItems = self.numItems + 1
        newItem = Item(key)
        self.itemList[key] = newItem
        return newItem

    def removeItem(self,key):
        self.numItems = self.numItems - 1
        del self.itemList[key]        

    def getVertex(self,n):
        if n in self.vertList:
            return self.vertList[n]
        else:
            return None

    def __contains__(self,n):
        return n in self.vertList

    def addEdge(self,f,t,cost=0):
        if f not in self.vertList:
            nv = self.addVertex(f)
        if t not in self.vertList:
            nv = self.addVertex(t)
        self.vertList[f].addOutgoingNeighbor(self.vertList[t], cost)
        self.vertList[t].addIncomingNeighbor(self.vertList[f], cost)

    def getVertices(self):
        return self.vertList.keys()

    def getItems(self):
        return self.itemList.keys()

    def __iter__(self):
        return iter(self.vertList.values())

def test():
    g = Graph()
    for i in range(10):
        g.addVertex(i)

    #0,1,2,3 are connected well to 5,6
    g.addEdge(0,5,5)
    g.addEdge(0,6,2)
    g.addEdge(1,5,4)
    g.addEdge(2,6,9)
    g.addEdge(2,9,1)
    g.addEdge(3,5,7)
    g.addEdge(3,6,3)

    low = g.addItem("low")
    low.associateVertex(g.getVertex(0))
    low.associateVertex(g.getVertex(1))
    low.associateVertex(g.getVertex(2))
    low.associateVertex(g.getVertex(3))

    mid = g.addItem("mid")
    mid.associateVertex(g.getVertex(5))
    mid.associateVertex(g.getVertex(6))

    high = g.addItem("high")
    high.associateVertex(g.getVertex(8))
    high.associateVertex(g.getVertex(9))

    g.addItem("other")

    mid.getPercentageOverlap(low)
    mid.getPercentageOverlap(low,threshold=30)
    high.getPercentageOverlap(low)
    high.getPercentageOverlap(low,threshold=0.5)

def generate_test(neurons=10,r=5,d=10):
    g = Graph()
    for i in range(neurons):
        g.addVertex(i)

    vertices = g.getVertices()

    for vertex in vertices:
        current_id = vertex
        #for random_neighbor in np.random.choice(vertices, d):
        for random_neighbor in sample_r_randomly(vertices, d):
            g.addEdge(vertex,random_neighbor,1)

    group1 = g.addItem("group1")
    #group1_vertices = np.random.choice(vertices, r)
    group1_vertices = sample_r_randomly(vertices, r)
    for vertice in group1_vertices:
        group1.associateVertex(g.getVertex(vertice))
    
    group2 = g.addItem("group2")
    #group2_vertices = np.random.choice(vertices, r)
    group2_vertices = sample_r_randomly(vertices, r)
    for vertice in group2_vertices:
        group2.associateVertex(g.getVertex(vertice))

    return g, vertices, group1, group2

def sample_r_randomly(vertices, r):
    idx = np.random.choice(len(vertices), r)   
    #return vertices[idx]
    return [vertices[i] for i in idx]

#given a graph, creates two random items and tests the overlap
#between them. Returns an array of n such overlap
def test_overlap(graph,r,n=1):
    overlaps = [0 for i in range(n)]
    vertices = graph.getVertices()

    for i in range(n):
        #generate random items
        group1 = graph.addItem("group1")

        #np.random.choice doesn't work over lists of tuples, so use this alternate
        #group1_vertices = np.random.choice(vertices, r)
        group1_vertices = sample_r_randomly(vertices, r)
        for vertice in group1_vertices:
            group1.associateVertex(graph.getVertex(vertice))
        
        group2 = graph.addItem("group2")
        #group2_vertices = np.random.choice(vertices, r)
        group2_vertices = sample_r_randomly(vertices, r)
        for vertice in group2_vertices:
            group2.associateVertex(graph.getVertex(vertice))

        #check the overlap of these two random items
        overlaps[i] = group1.getPotentialOverlap(group2)

        graph.removeItem("group1")
        graph.removeItem("group2")

    return overlaps



#g, vertices, group1, group2 = graph.generate_test(neurons=100,r=25,d=10)
#graph.test_overlap(group2, group1)
#group1.associateWith(group2)
#graph.test_overlap(group2, group1)
