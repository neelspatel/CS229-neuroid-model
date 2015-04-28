import cPickle as pickle
import gen_graph
import simple_graph

size_and_degree = {}

sizes = [150, 200, 250, 300]
degrees = [32, 64, 128, 256, 512, 1024]

for size in sizes:
	for degree in degrees:
		current_graph = gen_graph.read_test_graph(size, degree)
		size_and_degree[(size, degree)] = simple_graph.testPotentialOverlap(graph=current_graph, l = size, w = size, degree = degree, r = size * size * .25, n = 1000)


size = 250
degree = 128
r_results = {}
current_graph = gen_graph.read_test_graph(size, degree)
for r in [float(x) * 0.1 for x in range(1,10)]:
	results = simple_graph.testPotentialOverlap(graph=current_graph, l = size, w = size, degree = degree, r = r, n = 100)
	r_results[r] = results
	print r, sum(results) / len(results)