import cPickle as pickle
import gen_graph
import simple_graph
import time

size_and_degree = {}

#sizes = [150, 200, 250, 300]
sizes = [150, 200]
appropriate_rs = [0,0,0,0]
appropriate_ds = []
#degrees = [32, 64, 128, 256, 512, 1024]
degrees = [32, 64, 128, 256]

with open("size_degree_r.txt", "w+") as outfile:
	for size in sizes:
		for degree in degrees:
			try:
				current_graph = gen_graph.read_test_graph(size, degree)

				for r in [float(x) * 0.01 for x in range(1,20)]:
			
					try:
						results = simple_graph.testPotentialOverlap(graph=current_graph, l = size, w = size, degree = degree, r = size * size * r, n = 100)
						size_and_degree[(size, degree, r)] = results
						
						row_to_write = map(str,[size, degree, r] + results)

						outfile.write("\t".join(row_to_write))
						outfile.write("\n")
					except:
						print "Error on", size, degree, r
			except:
				print "Error on", size, degree, r

			


