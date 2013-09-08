import numpy as np
from time import time
import _mcqd as mcqd
import itertools
from logging import debug, warning

class CorrGraph(object):
    """Correspondence graph"""
    def __init__(self, options, sub_graph):
        """Takes in a sub_graph"""
        self.options = options
        self.sub_graph = sub_graph
        self.size = 0
        self.edge_count = 0
        self.nodes = None
        self.adj_matrix = None


    def extract_clique(self):
        debug("Commencing maximum clique")
        t1 = time()
        mc = mcqd.maxclique(np.array(self.adj_matrix, copy=True, order="C", dtype=np.int32), self.size)
        t2 = time() - t1
        debug("clique found, length of clique = %i, time reports %f seconds"%(
            len(mc), t2))
        return mc

    @property
    def pair_graph(self):
        self._pair_graph = None

    @pair_graph.setter
    def pair_graph(self, sub_graph):
        self._pair_graph = sub_graph

    @pair_graph.getter
    def pair_graph(self):
        return self._pair_graph

    @pair_graph.deleter
    def pair_graph(self):
        self.adj_matrix = None
        del self._pair_graph

    # this is really slow. Maybe implement in c++?
    def correspondence(self):
        self.edge_count = 0
        sub1 = self.sub_graph
        sub2 = self._pair_graph
        debug("Size of base sub graph = %i"%len(self.sub_graph))
        debug("Computing correspondence graph with %s"%sub2.name)
        t1 = time()
        nodes = [x for x in 
                itertools.product(range(len(sub1)), range(len(sub2)))
                if sub1[x[0]] == sub2[x[1]]]
        self.nodes = [(ind, x[0], x[1]) for ind, x in enumerate(nodes)]
        self.size = len(self.nodes)
        self.adj_matrix = np.zeros((self.size, self.size), dtype=np.int32)
        node_pairs = itertools.combinations(self.nodes, 2)
        for (n1, n11, n21),(n2, n12, n22) in node_pairs:
            if abs(sub1.distances[n11][n12] - 
                    sub2.distances[n21][n22]) <= self.options.tolerance:
                
                self.edge_count += 1
                self.adj_matrix[n1][n2] = 1
                self.adj_matrix[n2][n1] = 1
        t2 = time() - t1
        try:
            debug("Correspondence completed after %f seconds"%t2)
            debug("Size of correspondence graph = %i"%(self.size))
            debug("Edge density = %f"%(float(self.edge_count)*2./
                                    (float(self.size)*(float(self.size)-1))))
        except ZeroDivisionError:
            return
            #warning("No correspondence graph could be generated for %s"%sub2.name)

    def __getitem__(self, i):
        """Return the indices of the first subgraph from the
        output of maxclique graph."""
        return self.nodes[i][1]

