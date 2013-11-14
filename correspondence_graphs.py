import numpy as np
#from memory_profiler import profile
from time import time
import sys
import itertools
import _mcqd as mcqd
from collections import Counter
from logging import debug, warning

#np.set_printoptions(threshold='nan')
class CorrGraph(object):
    """Correspondence graph"""
    def __init__(self, options=None, sub_graph=None):
        """Takes in a sub_graph"""
        self.options = options
        self.sub_graph = sub_graph
        self.edge_count = 0
        self.nodes = None
        self.adj_matrix = None

    def extract_clique(self):
        while 1:
            debug("Commencing maximum clique")
            t1 = time()
            mc = mcqd.maxclique(self.adj_matrix, 
                                self.size)
            clique = [self[i] for i in mc] 
            # set adj_matrix entries to zero.
            zero_inds = self.get_adj_indices(mc)
            #self.set_to_zero(zero_inds)
            self.delete_inds(zero_inds)
            t2 = time() - t1
            debug("clique found, length of clique = %i, "%(len(mc)) + 
                  "time reports %f seconds"%(t2))

            yield clique 

    def get_adj_indices(self, mc):
        """Due to the simple way in which the correspondence graph is
        constructed, the range of nodes corresponding to the nodes of the
        sub_graph are easily detectable."""
        indices = [self[i] for i in mc]
        adj_indices = []
        # expand indices based on the range of pairings
        for i in indices:
            adj_indices += [j for j in range(self.size) if self[j] == i]
        return adj_indices

    def set_to_zero(self, inds):
        #set rows to zero
        self.adj_matrix[inds] = self.adj_matrix[inds].clip(max=0)
        #set cols to zero
        for i in range(self.size):
            self.adj_matrix[i][inds] = self.adj_matrix[i][inds].clip(max=0)

    def delete_inds(self, inds):
        #set rows to zero
        try:
            self.adj_matrix = np.delete(self.adj_matrix, inds, 0)
            #set cols to zero
            self.adj_matrix = np.delete(self.adj_matrix, inds, 1)
            self.nodes = np.delete(self.nodes, inds, 0)
        except (MemoryError, IndexError) as e:
            self.nodes = []

            self.adj_matrix = np.zeros((3,3))
        self.edge_count = np.count_nonzero(self.adj_matrix)

    @property
    def size(self):
        return len(self.nodes)

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
    def correspondence(self, tol=0.1):
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
        self.adj_matrix = np.zeros((self.size, self.size), dtype=np.int32)
        node_pairs = itertools.combinations(self.nodes, 2)
        try:
            tolerance = self.options.tolerance
        except AttributeError:
            tolerance = tol
        for (n1, n11, n21),(n2, n12, n22) in node_pairs:
            if abs(sub1.distances[n11][n12] - 
                    sub2.distances[n21][n22]) <= tolerance:
                
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
    
    def correspondence_api(self, tol=0.1):

        sub1 = self.sub_graph
        sub2 = self._pair_graph
        debug("Size of base sub graph = %i"%len(self.sub_graph))
        debug("Computing correspondence graph with %s"%sub2.name)
        t1 = time()
        try:
            tolerance = self.options.tolerance
        except AttributeError:
            tolerance = tol

        # check for correspondence size if asked in the input
        if self.options.max_correspondence:
            # compute the size of the correspondence graph based on 
            # the similar elements.
            size = self.compute_correspondence_size(sub1.elements,
                                                    sub2.elements)
            if size > self.options.max_correspondence:
                debug("The size of the correspondence graph is %i,"%(size) +
                        " which is greater than %i, so it will not be calculated"
                        %(self.options.max_correspondence))
                self.adj_matrix = np.zeros((3,3), dtype=np.int32)
                self.nodes = []
                return

        self.nodes = mcqd.correspondence(sub1.elements, sub2.elements)
        if len(self._pair_graph) > 1:
            try:
                self.adj_matrix = mcqd.correspondence_edges(self.nodes, 
                                      sub1.distances,
                                      sub2.distances,
                                      tolerance)
            except MemoryError:
                self.adj_matrix = np.zeros((3,3), dtype=np.int32)
        else:
            self.adj_matrix = np.zeros((self.size, self.size), dtype=np.int32)

        if self.adj_matrix is None:
            # catch memory error
            return
        #self.adj_matrix = np.array(adj_matrix, dtype=np.int32)
        self.nodes = [(i, j, k) for i, (j,k) in enumerate(self.nodes)]
        self.edge_count = np.count_nonzero(self.adj_matrix)
        t2 = time() - t1
        try:
            debug("Correspondence completed after %f seconds"%t2)
            debug("Size of correspondence graph = %i"%(self.size))
            debug("Edge density = %f"%(float(self.edge_count)*2./
                                    (float(self.size)*(float(self.size)-1))))
        except ZeroDivisionError:
            return
    
    def compute_correspondence_size(self, elem1, elem2):
        # create dic keys and counts
        e1, e2 = Counter(elem1), Counter(elem2)
        result = 0
        for key, e1val in e1.iteritems():
            e2val = e2.get(key, None)
            if e2val:
                result += e2val*e1val
        return result

    def __getitem__(self, i):
        """Return the indices of the first subgraph from the
        output of maxclique graph."""
        return self.nodes[i][1]

