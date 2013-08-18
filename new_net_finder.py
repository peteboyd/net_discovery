#!/usr/bin/env python
import options
import sys
import os
import numpy as np
import itertools
import ConfigParser
from scipy.spatial import distance
from time import time
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
        self.nodes = None
        self.elements = []
        self._coordinates = None 

    def from_faps(self, struct):
        cell = struct.cell.cell
        inv_cell = np.linalg.inv(cell.T)
        size = len(struct.atoms)
        # number of cells 
        multiplier = reduce(operator.mul, self.options.supercell, 1)
        self.nodes = np.array(range(size*multiplier))
        self._coordinates = np.empty((size*multiplier, 3), dtype=np.float64)
        self.elements = range(size*multiplier)
        for id, atom in enumerate(struct.atoms):
            supercell = itertools.product(*[itertools.product(range(j)) for j in
                self.options.supercell])
            # create 3x3x3 supercell to make sure all molecules are found
            for mult, scell in enumerate(supercell):
                self.elements[id + mult * size] = atom.element    
                fpos = atom.ifpos(inv_cell) + np.array([float(i[0]) for i in scell])
                self._coordinates[id + mult * size][:] = np.dot(fpos, cell)

    def from_sbu(self, sbu):
        size = len(sbu.atoms)
        self.nodes = np.array(range(size))
        self._coordinates = np.empty((size, 3), dtype=np.float64)
        for id, atom in enumerate(sbu.atoms):
            self.elements.append(atom.element)
            self._coordinates[id][:] = atom.coordinates[:3]

    @property
    def distances(self):
        try:
            return self._dmatrix
        except AttributeError:
            self._dmatrix = distance.cdist(self._coordinates, self._coordinates)
            return self._dmatrix


class CorrGraph(object):
    """Correspondence graph"""
    def __init__(self, sub1, sub2, options):
        """Takes in two subgraphs, creates a correspondence graph."""
        self.options = options
        self.size = 0
        self.edge_count = 0
        self.nodes = None
        self.adj_matrix = None
        self._gen_correspondence(sub1, sub2)

    def _gen_correspondence(self, sub1, sub2):
        nodes = [i for i in 
                itertools.product(sub1.nodes, sub2.nodes)
                if sub1.elements[i[0]] == sub2.elements[i[1]]]
        self.nodes = [(ind, i[0], i[1]) for ind, i in enumerate(nodes)]
        size = len(nodes)
        self.size = size
        self.adj_matrix = np.zeros((size, size), dtype=np.int32)
        node_pairs = itertools.combinations(self.nodes, 2)
        for (n1, n11, n21),(n2, n12, n22) in node_pairs:
            if abs(sub1.distances[n11][n12] - 
                    sub2.distances[n21][n22]) <= self.options.tolerance:
                
                self.edge_count += 1
                self.adj_matrix[n1][n2] = 1
                self.adj_matrix[n2][n1] = 1

    def get_sub_index(self, array):
        """Returs the indices of the first subgraph from the
        correspondence graph."""
        return [self.nodes[j][1] for j in array] 

    def pop(self, array):
        """Remove rows/columns from adj_matrix"""
        assert self.adj_matrix.any()
        # reverse sort the array to pop the right rows/columns
        array = reversed(sorted(list(array)))
        for x in array:
            self.size -= 1
            self.nodes.pop(x)
            # remove the column
            self.adj_matrix = np.delete(self.adj_matrix, (x), axis=1)
            # remove the row
            self.adj_matrix = np.delete(self.adj_matrix, (x), axis=0)

class ExtractSBUs(object):
    def __init__(self, mof, options):
        self.options = options
        self.mof = mof
        self.mof_graph = SubGraph(options, mof.name)
        self.mof_graph.from_faps(mof)
        # store correspondence graphs for repeated access
        self._corr_graphs = {}
        self._sbus = {}

    def gen_correspondence(self, sbu):
        """Generate correspondence graph with an sbu"""
        sbu_graph = SubGraph(self.options, sbu.name)
        sbu_graph.from_sbu(sbu)
        debug("Size of mof = %i"%(len(self.mof.atoms)))
        debug("Commencing correspondence graph with %s"%(sbu.name))
        t1 = time()
        c = CorrGraph(self.mof_graph, sbu_graph, self.options)
        t2 = time() - t1
        debug("Correspondence completed after %f seconds"%t2)
        debug("Size of correspondence graph = %i"%(c.size))
        debug("Edge density = %f"%(float(c.edge_count)*2./
                                    (float(c.size)*(float(c.size)-1))))
        self._corr_graphs[sbu.name] = c

    def extract_sbu(self, sbu):
        try:
            self._corr_graphs[sbu.name]
        except KeyError:
            self.gen_correspondence(sbu)
        C = self._corr_graphs[sbu.name]
        info("Commencing maximum clique")
        t1 = time()
        mc = mcqd.maxclique(C.adj_matrix.copy(), C.size)
        t2 = time() - t1
        info("clique found, length of clique = %i, time reports %f seconds"%(
            len(mc), t2))
        
        sbu_non_hydrogens = [i.element for i in sbu.atoms if i.element != "H"]
        sub1_indices = C.get_sub_index(mc)
        mof_non_hydrogens = [self.mof_graph.elements[i] for i in 
                             sub1_indices if self.mof_graph.elements[i] != "H"]
        if sorted(sbu_non_hydrogens) == sorted(mof_non_hydrogens):
            C.pop(mc)
        else:
            sub1_indices = []

        return sub1_indices

    def pop_mof_subgraph(self, array):
        """Removes nodes from the MOF subgraph, so they are
        not counted with future calculated correspondence graphs."""
        array = reversed(sorted(array))
        return [self.mof_graph.pop(x) for x in array]

    def read_sbu_files(self):
        for file in self.options.sbu_files:
            sbu_config = ConfigParser.SafeConfigParser()
            sbu_config.read(os.path.expanduser(file))
            for sbu_io in sbu_config.sections():
                sbu = SBU()
                sbu.from_config(sbu_io, sbu_config)
                name = "m" if sbu.is_metal else "o"
                name += str(sbu.identifier)
                name += "s" if sbu.parent is not None else ""
                self._sbus[name] = sbu

    def parse_groin_mofname(self):
        """metal, organic1, organic2, topology, functional group code"""
        ss = self.mof.name.split("_")
        met = ss[1]
        o1 = ss[2]
        o2 = ss[3]
        pp = ss[-1].split('.')
        top = pp[0]
        fnl = pp[2]
        return met, o1, o2, top, fnl

    def from_groin_mof(self):
        """Extract the sbus from groin mofs."""
        met, o1, o2, top, fnl = self.parse_groin_mofname()
        # get the sbus and functional groups, sort them
        # by length, then extract all the maximal cliques
        # above a certain value.
        sbu_list = []
        for i, sbu in self._sbus.items():
            if i.rstrip('s') in [met, o1, o2]:
                non_hydrogen_count = len([i for i in sbu.atoms 
                                      if i.element != "H"])
                sbu_list.append((non_hydrogen_count, sbu))
        sbu_list = [i[1] for i in reversed(sorted(sbu_list))]
        for sbu in sbu_list:
            info(sbu.name)
            info("Length of base graph: %i"%(
                len([i.element for i in sbu.atoms if i != "H"])))
            q = self.extract_sbu(sbu)
            while q:
                q = self.extract_sbu(sbu)

def main():
    mofs = CSV(options.csv_file)
    # read in sbus
    # generate distance matrices for the 'nodes' of the sbus

    for mof_name in mofs.get('MOFname'):
        info(mof_name)
        mof = Structure(mof_name)
        try:
            mof.from_file(os.path.join(options.lookup,
                         mof_name), "cif", '')
        except IOError:
            mof.from_file(os.path.join(options.lookup,
                          mof_name+".out"), "cif", '')
        ex = ExtractSBUs(mof, options)
        ex.read_sbu_files()
        ex.from_groin_mof()
        # generate distance matrix for the 'nodes' of the 3x3x3 mof
        # correspondence graph --> pair all nodes of the graph and the
        # sbu, create edges between nodes if their distance are the same?

if __name__=="__main__":
    main()
