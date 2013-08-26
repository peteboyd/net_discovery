#!/usr/bin/env python
import options
import sys
import os
import numpy as np
import itertools
import ConfigParser
from scipy.spatial import distance
from time import time
from copy import copy
import operator
from logging import info, debug, warning, error, critical
options = options.Options()
sys.path.append(options.faps_dir)
from faps import Structure
from function_switch import FunctionalGroupLibrary, FunctionalGroup
sys.path.append(options.genstruct_dir)
from SecondaryBuildingUnit import SBU
sys.path.append(options.max_clique_dir)
from plotter import GraphPlot
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
        self._new_index = []
        self.bonds = {}

    def from_faps(self, struct):
        cell = struct.cell.cell
        inv_cell = np.linalg.inv(cell.T)
        size = len(struct.atoms)
        # number of cells 
        multiplier = reduce(operator.mul, self.options.supercell, 1)
        self._coordinates = np.empty((size*multiplier, 3), dtype=np.float64)
        self.elements = range(size*multiplier)
        self._orig_index = range(size*multiplier)
        self._new_index = range(size*multiplier)
        supercell = list(itertools.product(*[itertools.product(range(j)) for j in
                    self.options.supercell]))
        
        for id, atom in enumerate(struct.atoms):

            for mult, scell in enumerate(supercell):
                # keep symmetry translated index
                self._orig_index[id + mult * size] = id
                self._new_index[id + mult * size] = id+mult*size
                self.elements[id + mult * size] = atom.element    
                fpos = atom.ifpos(inv_cell) + np.array([float(i[0]) for i in scell])
                self._coordinates[id + mult * size][:] = np.dot(fpos, cell)

        # shift the centre of atoms to the middle of the unit cell.
        supes = (cell.T * self.options.supercell).T
        isupes = np.linalg.inv(supes.T)
        self.debug("supercell")
        cou = np.sum(cell, axis=0)/2. 
        coa = np.sum(supes, axis=0)/2. 
        shift = coa - cou
        self._coordinates += shift
        # put all the atoms within fractional coordinates of the
        # supercell
        for i, cc in enumerate(self._coordinates):
            frac = np.array([k%1 for k in np.dot(isupes, cc)])
            self._coordinates[i] = np.dot(frac, supes)
        self.debug("supercell")
        # create supercell bond matrix
        for bond, val in struct.bonds.items():
            for mult, scell in enumerate(supercell):
                b1 = bond[0] + mult*size 
                b2 = bond[1] + mult*size 
                self.bonds[(b1,b2)] = val
        # shift the cell back??
        shift = cou - coa
        self._coordinates += shift
        self.debug("supercell")

    def compute_bonds(self):
        """Currently only implemented for the faps.Structure object"""
        for (b1, b2), val in self.bonds.items():
            # try other b2s, then other b1's
            b2_image = self._orig_index[b2]
            b2_images = [i for i, j in enumerate(self._orig_index) 
                         if j == b2_image]

            img2 = min([(self.distances[b1][i],i) for i in b2_images])[1]
            b1_image = self._orig_index[b1]
            b1_images = [i for i, j in enumerate(self._orig_index) 
                         if j == b1_image]
            img1 = min([(self.distances[b2][i],i) for i in b1_images])[1]
            if tuple(sorted([b1, img2])) not in self.bonds.keys():
                self.bonds.pop((b1, b2))
                self.bonds[tuple(sorted([b1, img2]))] = val
            elif tuple(sorted([b2, img1])) not in self.bonds.keys():
                self.bonds.pop((b1, b2))
                self.bonds[tuple(sorted([b2, img1]))] = val

    def from_fnl(self, obj):
        size = len(obj.atoms)
        self._coordinates = np.empty((size, 3), dtype=np.float64)
        self.elements = range(size)
        for id, atom in enumerate(obj.atoms):
            self._orig_index.append(id)
            self._new_index.append(id)
            self.elements[id] = atom.element
            self._coordinates[id][:] = atom.pos[:3]

    def from_sbu(self, sbu):
        size = len(sbu.atoms)
        self._coordinates = np.empty((size, 3), dtype=np.float64)
        self.elements = range(size)
        for id, atom in enumerate(sbu.atoms):
            self._orig_index.append(id)
            self._new_index.append(id)
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
        del self._new_index[x]
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
            sub._dmatrix[x][y] = self.distances[array[x]][array[y]]
        sub.elements = [self[x] for x in array]
        sub._coordinates = np.array([self._coordinates[x] for x in array],
                                    dtype=np.float64)
        sub._orig_index = [self._orig_index[x] for x in array]
        sub._new_index = [self._new_index[x] for x in array]
        sub.bonds = {(i1,i2):val for (i1, i2), val in self.bonds.items() if i1 in 
                        sub._new_index and i2 in sub._new_index}
        sub.bonds = self.bonds.copy()
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
        self._new_index += obj._new_index
        self.bonds.update(ojb.bonds)
        return self

    @property
    def distances(self):
        try:
            return self._dmatrix
        except AttributeError:
            self._dmatrix = distance.cdist(self._coordinates, self._coordinates)
            return self._dmatrix

    @property
    def centre_of_atoms(self):
        return np.average(self._coordinates, axis=0) 

    def debug(self, name="defile"):
        defile = open(name+".xyz", "a")
        defile.writelines("%i\n%s\n"%(len(self), name))
        for ind, (x, y, z) in enumerate(self._coordinates):
            defile.writelines("%s %12.5f %12.5f %12.5f\n"%(self[ind], x, y, z))
        defile.close()

class OrganicSBU(SubGraph):
 
    def to_mol(self):
        """Convert fragment to a string in .mol format"""
        header = "Organic\n %s\n\n"%(self.name)
        counts = "%3i%3i%3i%3i%3i%3i%3i%3i%3i 0999 V2000\n"
        atom_block = "%10.4f%10.4f%10.4f %3s%2i%3i%3i%3i%3i%3i%3i%3i%3i%3i%3i%3i\n"
        bond_block = "%3i%3i%3i%3i%3i%3i%3i\n"
        mol_string = header
        mol_string += counts%(len(self), len(self.bond.keys()), 0, 0, 0, 0, 0, 0, 0)
        atom_order = []
        for i in range(len(self)):
            pos = self._coordinates[i]
            mol_string += atom_block%(pos[0], pos[1], pos[2], self.elements[i])
            atom_order.append(self._new_index[i])

        for bond, type in self.bond.items():
            ind1 = atom_order.index(bond[0]) + 1
            ind2 = atom_order.index(bond[1]) + 1
            b_order = 4 if type == 1.5 else type
            mol_string += bond_block%(ind1, ind2, b_order, 0, 0, 0, 0)
        return mol_string

    def obtain_coordinating_groups(self, coord_groups, graph):
        """Takes a graph of SBU fragments as arguments and searches for 
        bonding coord groups with itself. These bonding groups are
        then added."""
        # search the graph edges for bonds with metalic SBUs

        # once found, find all coordinating groups of a specific kind

        # the coordinating groups are then appended here.

    def obtain_coordinating_functional_groups(self, graph):
        """Takes a graph of SBU fragments as arguments and searches
        for bonding with functional groups. These functional groups are
        then added to this SBU."""
        pass

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
            warning("No correspondence graph could be generated for %s"%sub2.name)

    def __getitem__(self, i):
        """Return the indices of the first subgraph from the
        output of maxclique graph."""
        return self.nodes[i][1]

class FunctionalGroups(object):
    
    def __init__(self, options):
        self.options = options
        self._groups = {}
        self._group_lookup = {}
        self._get_functional_subgraphs()
        if not options.sql_file:
            warning("No SQL database file found, the clique algorithm" +
                    " will iterate through all functional groups in the"+
                    " faps database, this may result in errors in the "+
                    "structure and take a VERY long time.")
        else:
            self._get_functional_group_lookup()

    def _get_functional_subgraphs(self):
        fnl_lib = FunctionalGroupLibrary()
        for name, obj in fnl_lib.items():
            if name != "H":
                sub = SubGraph(self.options, name)
                sub.from_fnl(obj)
                self._groups[name] = sub

    def _get_functional_group_lookup(self):
        """Read in all the mofs and store in the dictionary."""
        path = os.path.abspath(self.options.sql_file)
        if not os.path.isfile(path):
            error("could not find the file: %s"%(self.options.sql_file))
            sys.exit(1)
        filestream = open(path, 'r')
        for line in filestream:
            line = line.split("|")
            mof = "%s.sym.%s"%(line[0], line[1])
            # use a dictionary to sort the functionalizations
            groups = line[2]
            dic = {}
            if len(groups) > 0:
                [dic.setdefault(i.split("@")[0], []).append(i.split("@")[1])
                        for i in groups.split(".")]
                try:
                    dic.pop("H")
                except KeyError:
                    pass
                fnl = dic.keys()
            else:
                fnl = []
            self._group_lookup[mof] = fnl
        filestream.close()

    def get_functional_groups(self, mofname):
        try:
            return self._group_lookup[mofname]
        except KeyError:
            warning("No functional groups found for %s"%mofname)
            return [self._groups.keys()]

    def __getitem__(self, mofname):
        return [self._groups[group] for group in 
                self._group_lookup[mofname]]

class Net(object):

    def __init__(self, options, mof):
        """Convert a faps Structure object into a net."""
        self.options = options
        self.mof = mof
        self.cell = mof.cell.cell
        self.icell = np.linalg.inv(mof.cell.cell.T)
        self.fragments = []
        self.nodes = []
        self.edge_vectors = None
        self.edge_matrix = None
        self.main_sub_graph = SubGraph(self.options, self.mof.name)
        if options.mofs_from_groin:
            self.main_sub_graph.from_faps(mof)
            self.main_sub_graph.compute_bonds()

    def get_groin_sbus(self, sbus, mofname):
        met, o1, o2, top, fnl = self.parse_groin_mofname(mofname)
        sbu_list = []
        for i, sbu in sbus.items():
            if i.rstrip('s') in [met, o1, o2]:
                sbu.name = i
                non_hydrogen_count = len([i for i in sbu.atoms 
                                      if i.element != "H"])
                sbu_list.append((non_hydrogen_count, sbu))
        return [i[1] for i in reversed(sorted(sbu_list))]

    def from_groin_mof(self, sbus, fnls):
        """Extract the sbus from groin mofs."""
        debug("Size of mof = %i"%(len(self.mof.atoms)))
        sbu_list = self.get_groin_sbus(sbus, self.mof.name)
        sub_graph_copy = self.main_sub_graph % range(len(self.main_sub_graph))
        clq = CorrGraph(self.options, sub_graph_copy)
        # get the sbus and functional groups, sort them
        # by length, then extract all the maximal cliques
        # above a certain value.

        for fnl in fnls:
            clq.pair_graph = fnl
            generator = self.gen_cliques(clq, NO_H=False)
            for clique in generator:
                self.fragments.append(clique)
                clique.debug("fnls")

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
                self.fragments.append(clique)
                clique.debug("sbus")
        debug("There remains %i un-attached nodes. These will"%(len(clq.sub_graph)) +
                " be ignored as they are considered fragments outside "+
                "the periodic boundaries.")
        self.get_dangling_hydrogens(clq.sub_graph)

    def evaluate_completeness(self):
        """Check against the original MOF to ensure that all atoms
        are associated with a particular fragment (clique sub-graph)"""
        original = range(len(self.mof.atoms))
        fragments = set([i for j in self.fragments for i in j._orig_index])
        return original == sorted(list(fragments))

    def get_dangling_hydrogens(self, sub_graph):
        for frag in self.fragments:
            extra_bonds = [i[1] for i
                            in itertools.product(frag._new_index, sub_graph._new_index)
                            if tuple(sorted(list(i))) in self.main_sub_graph.bonds.keys()] 
            for k in extra_bonds:
                sub_index = sub_graph._new_index.index(k)
                if sub_graph[sub_index] == "H":
                    frag += (sub_graph % [sub_index])

    def get_edges(self):
        """Evaluate if an edge connects two fragments."""
        # shift the centre of atoms of the main_sub_graph to the 
        # centre of the cell.
        frag_pairs = itertools.combinations([(i,frag) for i, frag in enumerate(self.fragments)], 2)
        self.edge_vectors = [] 
        self.edge_matrix = np.zeros((0, len(self.fragments)), dtype=np.int32)
        for (ind1, frag1), (ind2, frag2) in frag_pairs:
            if self.edge_exists(frag1, frag2):
                edge_vector = frag1.centre_of_atoms - \
                        frag2.centre_of_atoms
                # create edge
                edge = np.zeros(len(self.fragments), dtype=np.int32)
                edge[ind1] = 1
                edge[ind2] = 1
                self.edge_matrix = np.vstack((self.edge_matrix, edge))
                self.edge_vectors.append((frag2.centre_of_atoms, edge_vector))

    def get_nodes(self):
        for frag in self.fragments:
            self.nodes.append(frag.centre_of_atoms)

    def prune_unit_cell(self):
        """Keep only those nodes which reside in the unit cell.
        Remove edges outside the unit cell."""
        # This is disgusting code.
        images = []
        fractionals = []
        for id, node in enumerate(self.nodes):
            frac = np.dot(self.icell, node)
            fractionals.append(frac)
            if any(np.logical_or(frac>1., frac<-0.0001)):
                images.append(id)
        # find image in unit cell
        unit_cells = [i for i in range(len(self.nodes)) if i not in images] 
        correspondence = range(len(self.nodes))
        node_shifts = []
        for node in images:
            f = fractionals[node]
            min_img = np.array([i%1 for i in f])
            node_img = [i for i in unit_cells if np.allclose(fractionals[i], 
                        min_img, atol=1e-4)]
            if len(node_img) == 0:
                warning("Could not find the image for one of the fragments!")
                # append the node?
                #self.nodes[node] = np.dot(min_img, self.cell)
                #node_shifts.append(images.index(node))
            elif len(node_img) > 1:
                warning("Multiple images found in the unit cell!!!")
            else:
                image = node_img[0]
                correspondence[node] = image

        # remove those nodes which needed to be shifted to the unit cell
        for i in reversed(sorted(node_shifts)):
            images.pop(i)
        # finally, adjust the edge matrix to correspond to the minimum image
        edge_pop = []
        for edge_id, edge in enumerate(self.edge_matrix):
            ids = [id for id, i in enumerate(edge) if i]
            if all([i in images for i in ids]):
                edge_pop.append(edge_id)
            else:
                for ii in ids:
                    edge[correspondence[ii]] = 1
        # now remove unnecessary edges, nodes.
        edge_pop = list(set(edge_pop))
        for xx in reversed(sorted(images)):
            self.nodes.pop(xx)
            self.fragments.pop(xx)
            self.edge_matrix = np.delete(self.edge_matrix, xx, axis=1)
        for xy in reversed(sorted(edge_pop)):
            self.edge_matrix = np.delete(self.edge_matrix, xy, axis=0)
            self.edge_vectors.pop(xy)

    def edge_exists(self, sub1, sub2):
        return any([tuple(sorted(list(i))) in self.main_sub_graph.bonds.keys() for i
            in itertools.product(sub1._new_index, sub2._new_index)]) 

    def gen_cliques(self, clq, NO_H=True):
        """The maxclique algorithm is non-discriminatory about the types
        of nodes it selects in the clique, therefore one SBU could be found
        in a maxclique from another SBU in the MOF.  To take measures against
        this, these cliques need to be ignored.  They are therefore
        removed, and re-instated after the algorithm has extracted all
        max cliques."""
        done = False
        if NO_H:
            compare_elements = clq.pair_graph.get_elements()
        else:
            compare_elements = clq.pair_graph.get_elements(NO_H=False)

        replace = []
        while not done:
            clq.correspondence()
            mc = clq.extract_clique()
            sub_nodes = sorted([clq[i] for i in mc])
            # get elements from sub_graph
            if NO_H:
                elem = clq.sub_graph.get_elements(sub_nodes)
            else:
                elem = clq.sub_graph.get_elements(sub_nodes,
                                                  NO_H=False)

            all_elem = clq.sub_graph.get_elements(sub_nodes, 
                                                  NO_H=False)
            # compare with elements from pair_graph
            if elem == compare_elements:
                clique = clq.sub_graph % sub_nodes
                clique.name = clq.pair_graph.name
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

    def show(self):
        gp = GraphPlot()
        gp.plot_cell(cell=self.cell, colour='g')
        for id, node in enumerate(self.nodes):
            name = self.fragments[id].name
            # metal == blue
            if name.startswith('m'):
                colour = 'b'
            # organic == green
            elif name.startswith('o'):
                colour = 'g'
            # functional group == red
            else:
                colour = 'r'
            gp.add_point(point=node, label=name, colour=colour)
     
        for ind, (point,edge) in enumerate(self.edge_vectors):
            # convert to fractional
            plot_point = np.dot(self.icell, point)
            plot_edge = np.dot(self.icell, edge)
            gp.add_edge(plot_edge, origin=plot_point)

        gp.plot()

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
    sbus = read_sbu_files(options)
    # read in functional groups
    fnls = FunctionalGroups(options)

    for mof_name in mofs.get('MOFname'):
        mof = Structure(mof_name)
        try:
            mof.from_file(os.path.join(options.lookup,
                         mof_name), "cif", '')
        except IOError:
            mof.from_file(os.path.join(options.lookup,
                          mof_name+".out"), "cif", '')
    
        ff = fnls[mof_name]
        net = Net(options, mof)
        if options.mofs_from_groin:
            clq = net.from_groin_mof(sbus, ff)
        else:
            error("NO implementation yet for non-groin MOFs.")
            sys.exit()
        if net.evaluate_completeness():
            net.get_edges()
            net.get_nodes()
            net.prune_unit_cell()
            net.show()
        # generate distance matrix for the 'nodes' of the 3x3x3 mof
        # correspondence graph --> pair all nodes of the graph and the
        # sbu, create edges between nodes if their distance are the same?

if __name__=="__main__":
    main()
