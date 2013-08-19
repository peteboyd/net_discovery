#!/usr/bin/env python
import options
import sys
import os
import numpy as np
import itertools
import ConfigParser
from scipy.spatial import distance
from time import time
from copy import deepcopy
import operator
from logging import info, debug, warning, error, critical
options = options.Options()
sys.path.append(options.faps_dir)
from faps import Structure
from function_switch import FunctionalGroupLibrary, FunctionalGroup
sys.path.append(options.genstruct_dir)
from SecondaryBuildingUnit import SBU
sys.path.append(options.max_clique_dir)
import _mcqd as mcqd


class CSV(object):
    """
    Reads in a .csv file for data parsing.

    """
    def __init__(self, filename):
        self.filename = filename
        self._data = {}
        if not os.path.isfile(filename):
            error("Could not find the file: %s"%filename)
            sys.exit(1)
        head_read = open(filename, "r")
        self.headings = head_read.readline().lstrip("#").split(",")
        head_read.close()
        self._read()

    def obtain_data(self, column, _TYPE="float", **kwargs):
        """return the value of the data in column, based on values of 
        other columns assigned in the kwargs.

        """
        matches = []
        # check to see if the columns are actually in the csv file
        for key, val in kwargs.items():
            if key not in self.headings:
                warning("Could not find the column %s in the csv file %s "%
                        (key, self.filename) + "returning...")
                return 0. if _TYPE is "float" else None
            matches += [i for i, j in enumerate(self._data[key]) if j == val]
        # warning: this is not exclusive.
        matches = set(matches)
        if len(matches) > 1:
            warning("Could not find a unique value for the data requested. "+
                    "Please refine your search with more column headings..")
            return 0. if _TYPE is "float" else None

        elif len(matches) == 0:
            warning("Could not find any match for the data requested!")
            return 0. if _TYPE is "float" else None

        return self._data[column][matches[0]]

    def _read(self):
        """The CSV dictionary will store data to heading keys"""
        filestream = open(self.filename, "r")
        burn = filestream.readline()
        for line in filestream:
            if not line.startswith("#"):
                line = line.split(",")
                for ind, entry in enumerate(line):
                    # convert to float if possible
                    try:
                        entry = float(entry)
                    except ValueError:
                        #probably a string
                        pass
                    self._data.setdefault(self.headings[ind], []).append(entry)
        filestream.close()

    def get(self, column):
        assert column in self._data.keys()
        return self._data[column]

class SubGraph(object):
    def __init__(self, options, name="Default"):
        self.options = options
        self.name = name
        self._coordinates = None
        self._orig_index = []

    def from_faps(self, struct):
        cell = struct.cell.cell
        inv_cell = np.linalg.inv(cell.T)
        size = len(struct.atoms)
        # number of cells 
        multiplier = reduce(operator.mul, self.options.supercell, 1)
        self._coordinates = np.empty((size*multiplier, 3), dtype=np.float64)
        self.elements = range(size*multiplier)
        self._orig_index = range(size*multiplier)
        supercell = list(itertools.product(*[itertools.product(range(j)) for j in
                    self.options.supercell]))
        for id, atom in enumerate(struct.atoms):
            for mult, scell in enumerate(supercell):
                # keep symmetry translated index
                self._orig_index[id + mult * size] = id
                self.elements[id + mult * size] = atom.element    
                fpos = atom.ifpos(inv_cell) + np.array([float(i[0]) for i in scell])
                self._coordinates[id + mult * size][:] = np.dot(fpos, cell)

    def from_sbu(self, sbu):
        size = len(sbu.atoms)
        self._coordinates = np.empty((size, 3), dtype=np.float64)
        self.elements = range(size)
        for id, atom in enumerate(sbu.atoms):
            self._orig_index.append(id)
            self.elements[id] = atom.element
            self._coordinates[id][:] = atom.coordinates[:3]
   
    @property
    def elements(self):
        self._elements = []

    @elements.setter
    def elements(self, val):
        self._elements = val

    @elements.getter
    def elements(self):
        return self._elements

    def get_elements(self, array=None, NO_H=True):
        if array is None:
            array = range(len(self))
        if NO_H:
            return sorted([self._elements[i] for i in array if 
                    self._elements[i] != "H"])
        else:
            return sorted([self._elements[i] for i in array])

    def __delitem__(self, x):
        del self.elements[x]
        del self._orig_index[x]
        self._dmatrix = np.delete(self._dmatrix, x, axis=0)
        self._dmatrix = np.delete(self._dmatrix, x, axis=1)
        self._coordinates = np.delete(self._coordinates, x, axis=0)

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, x):
        return self.elements[x]

    def __mod__(self, array):
        """Returns a SubGraph object with only the elements contained
        In array"""
        sub = SubGraph(self.options, name=self.name)
        size = len(array)
        sub._dmatrix = np.zeros((size, size), dtype=np.float64)
        for x, y in itertools.combinations(range(size), 2):
            sub._dmatrix[x][y] = self._dmatrix[array[x]][array[y]]
        sub.elements = [self[x] for x in array]
        sub._coordinates = np.array([self._coordinates[x] for x in array],
                                    dtype=np.float64)
        sub._orig_index = [self._orig_index[x] for x in array]
        return sub

    def __iadd__(self, obj):
        """Re-implements a Subgraph"""
        # Need to re-compute distances.
        try:
            del self._dmatrix
        except AttributeError:
            pass
        self.elements += obj.elements
        self._coordinates = np.vstack((self._coordinates, obj._coordinates))
        self._orig_index += obj._orig_index
        return self

    @property
    def distances(self):
        try:
            return self._dmatrix
        except AttributeError:
            self._dmatrix = distance.cdist(self._coordinates, self._coordinates)
            return self._dmatrix

    def debug(self, name="defile"):
        defile = open(name+".xyz", "a")
        defile.writelines("%i\n%s\n"%(len(self), name))
        for ind, (x, y, z) in enumerate(self._coordinates):
            defile.writelines("%s %12.5f %12.5f %12.5f\n"%(self[ind], x, y, z))
        defile.close()

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
        mc = mcqd.maxclique(self.adj_matrix.copy(), self.size)
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
        debug("Correspondence completed after %f seconds"%t2)
        debug("Size of correspondence graph = %i"%(self.size))
        debug("Edge density = %f"%(float(self.edge_count)*2./
                                    (float(self.size)*(float(self.size)-1))))

    def __getitem__(self, i):
        """Returs the indices of the first subgraph from the
        output of maxclique graph."""
        return self.nodes[i][1]

class Net(object):

    def __init__(self, options):
        """Convert a faps Structure object into a net."""
        self.options = options
    
    def get_groin_sbus(self, sbus, mofname):
        met, o1, o2, top, fnl = self.parse_groin_mofname(mofname)
        sbu_list = []
        for i, sbu in sbus.items():
            if i.rstrip('s') in [met, o1, o2]:
                non_hydrogen_count = len([i for i in sbu.atoms 
                                      if i.element != "H"])
                sbu_list.append((non_hydrogen_count, sbu))
        return [i[1] for i in reversed(sorted(sbu_list))]

    def from_groin_mof(self, mof, sbus):
        """Extract the sbus from groin mofs."""
        debug("Size of mof = %i"%(len(mof.atoms)))
        main_sub_graph = SubGraph(self.options, mof.name)
        main_sub_graph.from_faps(mof)
        sbu_list = self.get_groin_sbus(sbus, mof.name)
        clq = CorrGraph(self.options, main_sub_graph)
        # get the sbus and functional groups, sort them
        # by length, then extract all the maximal cliques
        # above a certain value.
        for sbu in sbu_list:
            sbu_cliques = []
            # this should be called from another function - no need
            # to keep re-calculating this sub_graph
            sbu_graph = SubGraph(self.options, sbu.name)
            sbu_graph.from_sbu(sbu)
            clq.pair_graph = sbu_graph
            generator = self.gen_cliques(clq)
            for clique in generator:
                # this line is why you need special identifiers for each node.
                clique.debug("sbus")

    def gen_cliques(self, clq):
        """The maxclique algorithm is non-discriminatory about the types
        of nodes it selects in the clique, therefore one SBU could be found
        in a maxclique from another SBU in the MOF.  To take measures against
        this, these cliques need to be ignored.  They are therefore
        removed, and re-instated after the algorithm has extracted all
        max cliques."""
        done = False
        compare_elements = clq.pair_graph.get_elements()
        replace = []
        while not done:
            clq.correspondence()
            mc = clq.extract_clique()
            sub_nodes = sorted([clq[i] for i in mc])
            # get elements from sub_graph
            elem = clq.sub_graph.get_elements(sub_nodes)
            all_elem = clq.sub_graph.get_elements(sub_nodes, 
                                                  NO_H=False)
            # compare with elements from pair_graph
            if elem == compare_elements:
                clique = clq.sub_graph % sub_nodes
                for xx in reversed(sub_nodes):
                    del clq.sub_graph[xx]
                #clq.sub_graph.debug()
                yield clique
            # in this instance we have found a clique not
            # belonging to the SBU. Remove, then replace
            elif len(all_elem) >= len(compare_elements):
                replace.append(clq.sub_graph % sub_nodes)
                for xx in reversed(sub_nodes):
                    del clq.sub_graph[xx]
            else:
                # reinsert false positive cliques
                for sub in replace:
                    clq.sub_graph += sub
                done = True

    def parse_groin_mofname(self, mof):
        """metal, organic1, organic2, topology, functional group code"""
        ss = mof.split("_")
        met = ss[1]
        o1 = ss[2]
        o2 = ss[3]
        pp = ss[-1].split('.')
        top = pp[0]
        fnl = pp[2]
        return met, o1, o2, top, fnl

def read_sbu_files(options):
    sbus = {}
    for file in options.sbu_files:
        sbu_config = ConfigParser.SafeConfigParser()
        sbu_config.read(os.path.expanduser(file))
        for sbu_io in sbu_config.sections():
            sbu = SBU()
            sbu.from_config(sbu_io, sbu_config)
            name = "m" if sbu.is_metal else "o"
            name += str(sbu.identifier)
            name += "s" if sbu.parent is not None else ""
            sbus[name] = sbu
    return sbus

def main():
    mofs = CSV(options.csv_file)
    # read in sbus
    # generate distance matrices for the 'nodes' of the sbus
    sbus = read_sbu_files(options)
    for mof_name in mofs.get('MOFname'):
        info(mof_name)
        mof = Structure(mof_name)
        try:
            mof.from_file(os.path.join(options.lookup,
                         mof_name), "cif", '')
        except IOError:
            mof.from_file(os.path.join(options.lookup,
                          mof_name+".out"), "cif", '')

        net = Net(options)
        if options.mofs_from_groin:
            net.from_groin_mof(mof, sbus)
        else:
            error("NO implementation yet for non-groin MOFs.")
            sys.exit()
        # generate distance matrix for the 'nodes' of the 3x3x3 mof
        # correspondence graph --> pair all nodes of the graph and the
        # sbu, create edges between nodes if their distance are the same?

if __name__=="__main__":
    main()
