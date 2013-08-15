#!/usr/bin/env python
import options
import sys
import os
import numpy as np
import itertools
import ConfigParser
from scipy.spatial import distance
from time import time
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
        self.nodes = np.array(range(size*27))
        self._coordinates = np.empty((size*27, 3), dtype=np.float64)
        self.elements = range(size*27)
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
        self.adj_matrix = None
        self._gen_correspondence(sub1, sub2)

    def _gen_correspondence(self, sub1, sub2):
        nodes = [i for i in 
                itertools.product(sub1.nodes, sub2.nodes)
                if sub1.elements[i[0]] == sub2.elements[i[1]]]
        nodes = [(ind, i[0], i[1]) for ind, i in enumerate(nodes)]
        size = len(nodes)
        self.size = size
        self.adj_matrix = np.zeros((size, size), dtype=np.int32)
        node_pairs = itertools.combinations(nodes, 2)
        for (n1, n11, n21),(n2, n12, n22) in node_pairs:
            if (sub1.distances[n11][n12] - 
                    sub2.distances[n21][n22]) <= self.options.tolerance:
                self.edge_count += 1
                self.adj_matrix[n1][n2] = 1
                self.adj_matrix[n2][n1] = 1

def read_sbu_file(filename):
    sbu_pool = []
    sbu_config = ConfigParser.SafeConfigParser()
    sbu_config.read(filename)
    for sbu_io in sbu_config.sections():
        sbu = SBU()
        sbu.from_config(sbu_io, sbu_config)
        sbu_pool.append(sbu)
    return sbu_pool

def main():
    mofs = CSV(options.csv_file)
    # read in sbus
    # generate distance matrices for the 'nodes' of the sbus

    sbus = read_sbu_file(os.path.expanduser(options.sbu_files[0]))
    for mof_name in mofs.get('MOFname'):
        info(mof_name)
        mof = Structure(mof_name)
        try:
            mof.from_file(os.path.join(options.lookup,
                         mof_name), "cif", '')
        except IOError:
            mof.from_file(os.path.join(options.lookup,
                          mof_name+".out"), "cif", '')
        q = SubGraph(options, sbus[1].name)
        q.from_sbu(sbus[1])
        p = SubGraph(options, mof.name)
        p.from_faps(mof)
        info("Size of mof = %i"%(len(mof.atoms)))
        info("Commencing correspondence graph")
        t1 = time()
        c = CorrGraph(q,p,options)
        t2 = time() - t1
        info("Correspondence completed after %f seconds"%t2)
        info("Size of correspondence graph = %i"%(c.size))
        info("Edge density = %f"%(float(c.edge_count)*2./(float(c.size)*(float(c.size)-1))))
        info("Commencing maximum clique")
        t1 = time()
        mc = mcqd.maxclique(c.adj_matrix, c.size)
        t2 = time() - t1
        info("DONE, length of clique = %i, time reports %f seconds"%(len(mc), t2))
        # generate distance matrix for the 'nodes' of the 3x3x3 mof
        # correspondence graph --> pair all nodes of the graph and the
        # sbu, create edges between nodes if their distance are the same?

if __name__=="__main__":
    main()
