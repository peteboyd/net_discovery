#!/usr/bin/env python

"""net_finder.py

This program was designed to scan through mass amounts of data to compile what
is deemed necessary information for each MOF.  This will include building units
where the functional group is located, uptake at given conditions, and other
such stuff.

"""
import options
from logging import info, debug, warning, error, critical
import itertools
import pickle
import math
import os
from uuid import uuid4
from scipy.spatial import distance
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import pybel
import copy
import time
from random import choice
#from mpi4py import MPI
#comm = MPI.COMM_WORLD
#rank = comm.Get_rank()
#size = comm.Get_size()
options = options.Options(rank=0)
sys.path.append(options.faps_dir)
from faps import Structure, Cell, Atom, Symmetry
from function_switch import FunctionalGroupLibrary, FunctionalGroup
from elements import CCDC_BOND_ORDERS
sys.path.append(options.genstruct_dir)
from genstruct import Database, BuildingUnit, Atom_gen
# 1. need to read in database of linkers
# 2. need to read in database of functional groups
# 3. need to read in sql file
# 4. associate H1 with a building unit
# 5. isolate molecules
# 6. inchikeys of molecules?
# 7. distortion of building unit?
# 8. need to compute neighbouring building units
# 9. need to represent reduced topology as a graph
# 10. assign info to the nodes
# 11. node is the COM of the building unit.. need to compute this.
class CSV(dict):
    """
    Reads in a .csv file for data parsing.

    """
    def __init__(self, filename, _MOFNAME=True):
        self._columns = {"MOF":"MOFname", "uptake":"mmol/g",
                      "temperature":"T/K", "pressure":"p/bar",
                      "heat of adsorption":"hoa/kcal/mol"}
        self.filename = filename
        if not os.path.isfile(filename):
            error("Could not find the file: %s"%filename)
            sys.exit(1)
        head_read = open(filename, "r")
        self.headings = head_read.readline().lstrip("#").split(",")
        head_read.close()
        if _MOFNAME:
            self._parse_by_mofname()
        else:
            self._parse_by_heading()

    def obtain_data(self, column, _TYPE="float", **kwargs):
        """return the value of the data in column, based on values of 
        other columns assigned in the kwargs.

        """
        # create a set of lists for the data we care about
        trunc = []
        trunc_keys = {} 
        # check to see if the columns are actually in the csv file
        for ind, key in enumerate([column] + kwargs.keys()):
            try:
                rightkey = self._columns[key]
            except KeyError:
                rightkey = key
            if rightkey not in self.headings:
                warning("Could not find the column %s in the csv file %s "%
                        (rightkey, self.filename) + "returning...")
                return 0. if _TYPE is "float" else None
            else:
                trunc.append(self[rightkey])
                trunc_keys[ind] = key
                if key == column:
                    colind = ind
    
        for entry in itertools.izip_longest(*trunc):
            # tie an entry list index to column + kwargs keys
            kwargs_id =[i for i in range(len(entry)) if trunc_keys[i] in 
                    kwargs.keys()]
            if all([entry[i] == kwargs[trunc_keys[i]] for i in kwargs_id]):
                # grab the entry for the column
                col = entry[colind]
                return float(col) if _TYPE is "float" else col

        warning("Didn't find the data point requested in the csv file %s"%
                self.filename)
        return 0. if _TYPE is "float" else None

    def _parse_by_heading(self):
        """The CSV dictionary will store data to heading keys"""
        filestream = open(self.filename, "r")
        # burn the header, as it's already read
        # if the file is empty append zeroes..
        if self._line_count(self.filename) <= 1: 
            for ind in range(len(self.headings)):
                self.setdefault(self.headings[ind], []).append(0.)
            filestream.close()
            return
        burn = filestream.readline()
        for line in filestream:
            if not line.startswith("#"):
                line = line.split(",")
                for ind, entry in enumerate(line):
                    try:
                        entry = float(entry)
                    except ValueError:
                        #probably a string
                        pass
                    self.setdefault(self.headings[ind], []).append(entry)
        filestream.close()

    def _line_count(self, filename):
        with open(filename) as f:
            for i, l in enumerate(f):
                pass
        return i + 1

    def _parse_by_mofname(self):
        """The CSV dictionary will have MOFnames as headings and contain
        sub-dictionaries for additional row data.

        """
        filestream = open(self.filename, "r")
        try:
            mofind = self.headings.index(self._columns["MOF"])
        except ValueError:
            error("the csv file %s does not have %s as a column! "%
                    (self.filename, self._columns["MOF"]) + 
                    "EXITING ...")
            sys.exit(0)

        try:
            uptind = self.headings.index(self._columns["uptake"])
        except ValueError:
            warning("the csv file %s does not have %s as a column"%
                    (self.filename, self._columns["uptake"]) +
                    " the qst will be reported as 0.0 kcal/mol")
        try:
            hoaind = self.headings.index(self._columns["heat of adsorption"])
        except ValueError:
            warning("the csv file %s does not have %s as a column"%
                    (self.filename, self._columns["heat of adsorption"]) +
                    " the qst will be reported as 0.0 kcal/mol")
        burn = filestream.readline()
        for line in filestream:
            line = line.strip()
            if line and not line.startswith("#"):
                line = line.split(",")
                mofname = line[mofind].strip()
                mofname = clean(mofname)
                try:
                    uptake = line[uptind]
                except UnboundLocalError:
                    uptake = 0.
                self.setdefault(mofname, {})["mmol/g"] = float(uptake)
                try:
                    hoa = line[hoaind]
                except UnboundLocalError:
                    hoa = 0.
                self.setdefault(mofname, {})["hoa"] = float(hoa)
        filestream.close()


class FunctionalGroups(dict):
    """
    Reads in a .sqlout file and returns a dictionary containing mofnames and
    their functionalizations.

    """

    def __init__(self, filename):
        """Read in all the mofs and store in the dictionary."""
        if not os.path.isfile(filename):
            error("could not find the file: %s"%(filename))
            sys.exit(1)
        filestream = open(filename, 'r')
        for line in filestream:
            line = line.split("|")
            mof = "%s.sym.%s"%(line[0], line[1])
            # use a dictionary to sort the functionalizations
            groups = line[2]
            dic = {}
            if len(groups) > 0:
                [dic.setdefault(i.split("@")[0], []).append(i.split("@")[1])
                        for i in groups.split(".")]
                if len(dic.keys()) == 1:
                    dic[None] = []
                elif len(dic.keys()) == 0:
                    dic[None] = []
                    dic[False] = []
            else:
                dic = {None:[], False:[]}
            # check if the entry already exists!
            if self._check_duplicate(mof):
                if self[mof] == dic:
                    # duplicate
                    debug("Duplicate found %s"%(mof))
                    #pass
                else:
                    warning("duplicate found for %s"%(mof) +
                            " but with different functionalization!")
            else:
                self[mof] = dic
        filestream.close()

    def _check_duplicate(self, entry):
        """Return true if the key exists, otherwise, false."""
        if self.has_key(entry):
            return True
        return False

class GraphPlot(object):
    
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.cell = np.identity(3)

    def plot_cell(self, cell, origin=np.zeros(3), colour='b'):
        # add axes labels
        self.cell = cell.copy()
        xyz_a = (cell[0]+origin)/2.
        xyz_b = (cell[1]+origin)/2.
        xyz_c = (cell[2]+origin)/2.
        self.ax.text(xyz_a[0], xyz_a[1], xyz_a[2], 'a')
        self.ax.text(xyz_b[0], xyz_b[1], xyz_b[2], 'b')
        self.ax.text(xyz_c[0], xyz_c[1], xyz_c[2], 'c')

        all_points = [np.sum(a, axis=0)+origin
                      for a in list(powerset(self.cell)) if a]
        all_points.append(origin)
        for s, e in itertools.combinations(np.array(all_points), 2):
            if any([zero_cross(s-e, i) for i in self.cell]):
                self.ax.plot3D(*zip(s, e), color=colour)
                
    def add_point(self, point=np.zeros(3), label=None, colour='r'):
        p = point.copy()
        p = np.dot(p, self.cell)
        try:
            self.ax.scatter(*p, color=colour)
        except TypeError:
            p = p.tolist()
            self.ax.scatter(p, color=colour)
        if label:
            self.ax.text(*p, s=label)
            
    def plot(self):
        plt.show()

    def add_edge(self, vector, origin=np.zeros(3), label=None, colour='y'):
        """Accounts for periodic boundaries by splitting an edge where
        it intersects with the plane of the boundary conditions.

        """
        point = origin - vector
        max = [0, 0]
        periodics = [(ind, mag) for ind, mag in enumerate(point)
                      if mag > 1. or mag < 0.]
        # the check should be if any of the values are greater than one or
        # less than zero
        #for ind, mag in enumerate(point):
        #    if abs(mag) > abs(max[1]):
        #        max = [ind, mag]
        # determine how many planes intersect the two points.
        if periodics:
            # temp fix
            max = periodics[0]
            # periodic boundary found
            # plane is defined by the other cell vectors
            plane_vec1, plane_vec2 = np.delete(self.cell, max[0], axis=0)
            # plane point is defined by the cell vector
            plane_pt = np.trunc(max[1]) * self.cell[max[0]]

            point1 = np.dot(origin, self.cell)
            vector1 = np.dot(vector, self.cell)
            point2 = point_of_intersection(point1, vector1, plane_pt,
                                           plane_vec1, plane_vec2)
            # periodic shift of point2
            point3 = point2 + np.floor(max[1])*-1 * self.cell[max[0]]
            # periodic shift of point
            point4 = np.dot(point - np.floor(point), self.cell)

            self.ax.plot3D(*zip(point2, point1), color=colour)
            self.ax.plot3D(*zip(point4, point3), color=colour)
        else:
            point1 = np.dot(origin, self.cell)
            point2 = np.dot(point, self.cell)
            self.ax.plot3D(*zip(point2, point1), color=colour)
        if label:
            p = origin + 0.5*vector
            p = p - np.floor(p)
            self.ax.text(*p, s=label)

    def _unused_example():
        theta = np.linspace(-4*np.pi, 4*np.pi, 100)
        z = np.linspace(-2,2,100)
        r = z**2 + 1
        x = r* np.sin(theta)
        y = r*np.cos(theta)
        ax.plot(x, y, z, label='parametric curve')
        ax.legend()
        plt.show()
        sys.exit()

def to_cif(atoms, cell, bonds, name, _DEBUG=False):
    """Return a CIF file with bonding and atom types."""

    inv_cell = cell.inverse

    type_count = {}
    bu_order = []
    bu_indices = {}
    atom_part = []
    for idx, atom in enumerate(atoms):
        if atom is None:
            # blanks are left in here
            continue
        if hasattr(atom, 'uff_type') and atom.uff_type is not None:
            uff_type = atom.uff_type
        else:
            uff_type = '?'
        if atom.element in type_count:
            type_count[atom.element] += 1
        else:
            type_count[atom.element] = 1
        atom.site = "%s%i" % (atom.element, type_count[atom.element])
        if _DEBUG:
            coord = atom.scaled_pos
            atmdescription = (atom.site, atom.element, uff_type, "?", "?")
        else:
            try:
                bu_type = "%i_%s"%(atom.molorder, atom.bu)
            except AttributeError:
                print "PROBLEMS WITH CIF WRITING"
                print atom.uff_type
            if bu_type in bu_order:
                buind = bu_order.index(bu_type)
            else:
                buind = len(bu_order)
                bu_order.append(bu_type)
                bu_indices[buind] = atom.bu
            if hasattr(atom, 'fnlorder'):
                fnl_type = "%i_%s"%(atom.fnlorder, atom.fnl)
                if fnl_type in bu_order:
                    fnlind = bu_order.index(fnl_type)
                else:
                    fnlind = len(bu_order)
                    bu_order.append(fnl_type)
                    bu_indices[fnlind] = atom.fnl
            else:
                fnlind = '?'

            coord = atom.ifpos(inv_cell)
            atmdescription = (atom.site, atom.element, uff_type, 
                              buind, fnlind)
        atom_part.append("%-5s %-5s %-5s %-5s %-5s" % atmdescription)
        atom_part.append("%f %f %f " % tuple(coord))
        atom_part.append("%f\n" % atom.charge)
    bond_part = []
    for bond, order in bonds.items():
        try:
            bond_part.append("%-5s %-5s %-5s\n" %
                             (atoms[bond[0]].site, atoms[bond[1]].site,
                              CCDC_BOND_ORDERS[order]))
        except AttributeError:
            # one of the atoms is None so skip
            debug("cif NoneType atom")

    cif_file = [
        "data_%s\n" % name.replace(' ', '_'),
        "%-33s %s\n" % ("_audit_creation_date", time.strftime('%Y-%m-%dT%H:%M:%S%z')),
        "%-33s %s\n" % ("_audit_creation_method", "Derpy_derp_derp"),
        "%-33s %s\n" % ("_symmetry_space_group_name_H-M", "P1"),
        "%-33s %s\n" % ("_symmetry_Int_Tables_number", "1"),
        "%-33s %s\n" % ("_space_group_crystal_system", cell.crystal_system),
        "%-33s %-.10s\n" % ("_cell_length_a", cell.a),
        "%-33s %-.10s\n" % ("_cell_length_b", cell.b),
        "%-33s %-.10s\n" % ("_cell_length_c", cell.c),
        "%-33s %-.10s\n" % ("_cell_angle_alpha", cell.alpha),
        "%-33s %-.10s\n" % ("_cell_angle_beta", cell.beta),
        "%-33s %-.10s\n" % ("_cell_angle_gamma", cell.gamma),
        "%-33s %s\n" % ("_cell_volume", cell.volume)]
    if not _DEBUG:
        bu_part = []
        for i in range(len(bu_indices.keys())):
            name = bu_indices[i]
            bu_part.append("%-5i %-12s\n"%(i, name))
        cif_file += [
                "\nloop_\n",
                "_chemical_identifier\n",
                "_chemical_name\n"] + bu_part
    # start of atom loops
    cif_file += [
        "\nloop_\n",
        "_atom_site_label\n",
        "_atom_site_type_symbol\n",
        "_atom_site_description\n",
        "_atom_site_building_unit\n",
        "_atom_site_functional_group\n",
        "_atom_site_fract_x\n",
        "_atom_site_fract_y\n",
        "_atom_site_fract_z\n",
        "_atom_type_partial_charge\n"] + atom_part 

    # bonding loop
    cif_file += [
        "\nloop_\n",
        "_geom_bond_atom_site_label_1\n",
        "_geom_bond_atom_site_label_2\n",
#        "_geom_bond_distance\n",
        "_ccdc_geom_bond_type\n"] + bond_part

    return cif_file

def point_of_intersection(p_edge, edge, p_plane, plane_vec1, plane_vec2):
    """
    Returns a point of intersection between an edge and a plane
    p_edge is a point on the edge vector
    edge is the vector direction
    p_plane is a point on the plane
    plane_vec1 represents one of the vector directions of the plane
    plane_vec2 represents the second vector of the plane

    """
    n = np.cross(plane_vec1, plane_vec2)
    n = n / np.linalg.norm(n)
    l = edge / np.linalg.norm(edge)
    
    ldotn = np.dot(l, n)
    pdotn = np.dot(p_plane - p_edge, n)
    if ldotn == 0.:
        return np.zeros(3) 
    if pdotn == 0.:
        return p_edge 
    return pdotn/ldotn*l + p_edge 

def cut_carboxylate_bridge(node, cif, cut_graph):
    """Deletes the C-C bond from a carboxylate bridging
    moiety.

    """
    neighbours = N(node, cut_graph)
    neighbour_types = [cut_graph[i]['element'] for i in neighbours]
    if ["C", "O", "O"] == sorted(neighbour_types):
        o_nodes = [i for i in neighbours if cut_graph[i]['element'] == "O"]
        extended_neighbours = [cut_graph[i]['element'] for j 
                               in o_nodes for i in N(j, cut_graph)]
        if not any([i == "H" for i in extended_neighbours]):
            c_node = [i for i in N(node, cut_graph) if
                      cut_graph[i]['element'] == "C"]
            c_node = c_node[0]
            c_ind = cut_graph[node]['neighbours'].index(c_node)
            cut_graph[node]['neighbours'].pop(c_ind)
            c_ind = cut_graph[c_node]['neighbours'].index(node)
            cut_graph[c_node]['neighbours'].pop(c_ind)
    return

def cut_pyrazole_bridge(node, cif, cut_graph):
    metals = ["Ni", "Mn"]
    neighbours = N(node, cut_graph)
    neighbour_types = [cut_graph[i]['element'] for i in neighbours]
    if "C" in neighbour_types:
        c_neighbours = [i for i in neighbours if 
                        cut_graph[i]['element'] == "C"]
        for cn in c_neighbours:
            pl = [i for i in N(cn, cut_graph) if i != node]
            ex = [j for i in pl for j in 
                  N(i, cut_graph) if j != cn]
            extended_neighbours = pl + ex
            if ["C", "C", "H", "H", "N", "N"] == sorted([cut_graph[i]['element'] 
                                        for i in extended_neighbours]):
                metal_test = [cut_graph[i]['element'] for j in 
                              extended_neighbours for 
                              i in N(j, cut_graph) if
                              cut_graph[j]['element'] == "N"]
                if any([i in metal_test for i in metals]):
                    c_ind = cut_graph[node]['neighbours'].index(cn)
                    cut_graph[node]['neighbours'].pop(c_ind)
                    c_ind = cut_graph[cn]['neighbours'].index(node)
                    cut_graph[cn]['neighbours'].pop(c_ind)

def cut_metal_nitrogen_bridge(node, cif, cut_graph):
    metals = ["Cu", "Zn", "In", "Cr", "Zr", "V"]
    neighbours = N(node, cut_graph)
    neighbour_types = [cut_graph[i]['element'] for i in neighbours]
    if any([i == j for i in metals for j in neighbour_types]):
        m_node = [i for i in N(node, cut_graph) if
                   cut_graph[i]['element'] in metals]
        m_node = m_node[0]
        m_ind = cut_graph[node]['neighbours'].index(m_node)
        cut_graph[node]['neighbours'].pop(m_ind)
        n_ind = cut_graph[m_node]['neighbours'].index(node)
        cut_graph[m_node]['neighbours'].pop(n_ind)
    return

def cut_phosphonate_bridge(node, cif, cut_graph):
    neighbours = N(node, cut_graph)
    c_node = [i for i in neighbours if cut_graph[i]['element'] == "C"]
    c_node = c_node[0]
    c_ind = cut_graph[node]['neighbours'].index(c_node)
    cut_graph[node]['neighbours'].pop(c_ind)
    p_ind = cut_graph[c_node]['neighbours'].index(node)
    cut_graph[c_node]['neighbours'].pop(p_ind)

def cut_mof_by_links(cif, graph, mofname):
    """Slices the bonds of MOFs at particular points to properly define
    molecular distances, and make the correspondence graph much faster
    to calculate.

    """
    inv = cif.cell.inverse
    cut_graph = copy.deepcopy(graph)
    met, org1, org2, top, fnum = parse_mof_name(mofname)
    for atom in cif.atoms:
        atom.scaled_pos = np.array(atom.ifpos(inv))
        node = atom.nodeid 
        if atom.type == 'C':
            if atom.uff_type == "C_R":
                cut_carboxylate_bridge(node, cif, cut_graph)
            if met == 'm12':
                cut_pyrazole_bridge(node, cif, cut_graph) 
        elif atom.type == "N":
            cut_metal_nitrogen_bridge(node, cif, cut_graph)
        elif atom.type == "P":
            cut_phosphonate_bridge(node, cif, cut_graph)
    # shift all by molecular periodic image
    node_recalc = []
    # some cheaty stuff here
    net_cut, slicey_nodes = [], []
    # end of cheaty stuff
    for node in cut_graph.keys():
        slicey_nodes = []
        if node not in node_recalc:
            for nnode in iter_neighbours(node, cut_graph):
                slicey_nodes.append(nnode)
                node_recalc.append(nnode)
            net_cut.append(slicey_nodes[:])
    # graph cut is just the graph where the bonds are sliced
    # net cut is one level of depth further, where each
    # index is a 'chunk' of the cif.
    return (cut_graph, net_cut)

def gen_cif_distance(cif, cut_graph, graph):
    """This computes the distances between molecular atoms.  These
    are established by 'cutting' the cif at intervals defined by a
    particular moiety.  In the initial case only carboxylates are 
    used.

    """
    inv = cif.cell.inverse
    node_recalc = []
    # shift adjacent nodes by the periodic image so that molecules
    # remain intact.
    for node in cut_graph.keys():
        for nnode in iter_neighbours(node, cut_graph):
            if nnode not in node_recalc:
                node_recalc.append(nnode)
                vals = cut_graph[nnode]
                atom = get_atom(nnode, cif)
                neighbours = N(nnode, cut_graph)
                for nodeid in neighbours:
                    adj_atom = get_atom(nodeid, cif)
                    diff = adj_atom.scaled_pos - atom.scaled_pos
                    adj_atom.scaled_pos -= np.round(diff)
    # now calculate distances.  The inter molecular distances will
    # be waaay off, but we don't really care about these.
    atom_pairs = list(itertools.combinations(cif.atoms, 2))
    for pair in atom_pairs:
        dist = pair[0].scaled_pos - pair[1].scaled_pos
        vect = np.dot(dist, cif.cell.cell)
        length = np.linalg.norm(vect)
        graph[pair[0].nodeid]['distance'][pair[1].nodeid] = length
        graph[pair[1].nodeid]['distance'][pair[0].nodeid] = length

    #lines = to_cif(cif.atoms, cif.cell, cif.bonds, "test", _DEBUG=True)
    #cif_name = "test.cif"
    #output_file = open(cif_name, "w")
    #output_file.writelines(lines)
    #output_file.close()

def gen_graph_faps(faps_obj):
    _graph = {}
    _bond_type = {}
    for idx, atom in enumerate(faps_obj.atoms):
        atom.idx = idx
        atom.nodeid = str(uuid4())
        _graph.setdefault(atom.nodeid, {})
        _graph[atom.nodeid]['neighbours'] = []
        _graph[atom.nodeid]['atomidx'] = idx
        _graph[atom.nodeid]['type'] = atom.uff_type
        _graph[atom.nodeid]['element'] = atom.type
        _graph[atom.nodeid]['distance'] = {}

    for bond, order in faps_obj.bonds.items():
        atomid1 = faps_obj.atoms[bond[0]].nodeid
        atomid2 = faps_obj.atoms[bond[1]].nodeid
        sortybond = sorted([atomid1, atomid2])
        _graph.setdefault(atomid1, {})
        _graph.setdefault(atomid2, {})
        _graph[atomid1]['neighbours'].append(atomid2)
        _graph[atomid2]['neighbours'].append(atomid1)
        _bond_type[tuple(sortybond)] = order 

    return _graph, _bond_type

def get_atom(nodeid, obj):
    for atom in obj.atoms:
        if nodeid == atom.nodeid:
            return atom
    return atom

def iter_neighbours(node, g):
    if node in g.keys():
        yield node
    queue = [i for i in N(node, g) if i in g.keys()]
    X = [node]
    while queue:
        yield queue[0]
        X.append(queue[0])
        queue = queue[1:] + N(queue[0], g)
        queue = [i for i in queue if i not in X and i in g.keys()]
                
def calc_com(chunk, obj):
    """Note, this is the NODE center of mass, not the
    molecular center of mass - it doesn't take into account
    atomic weights, each node is weighted the same (as 1).
    
    """
    
    if isinstance(obj, Structure) or isinstance(obj, FunctionalGroup):
        for node in iter_neighbours(chunk.keys()[0], chunk):
            vals = chunk[node]
            atom = get_atom(node, obj)
            # make sure not to shift adjacent nodes on other chunks
            neighbours = [i for i in vals['neighbours'] if i in chunk.keys()]
            for nodeid in neighbours:
                adj_atom = get_atom(nodeid, obj)
                diff = adj_atom.scaled_pos - atom.scaled_pos
                adj_atom.scaled_pos -= np.round(diff)
        # ignore the hydrogen atoms for the com calc
        positions = np.array([i.scaled_pos for i in obj.atoms
                              if i.nodeid in chunk.keys() and
                              not i.uff_type.startswith("H")])
        # note COM might not be in the unit cell, this must
        # be shifted, along with the nodes themselves.
        com = np.sum(positions, axis=0) / float(len(positions))
        com_shift = np.floor(com)

        for node in chunk.keys():
            atom = get_atom(node, obj)
            atom.scaled_pos -= com_shift

        return com - com_shift 
    
    elif isinstance(obj, FunctionalGroup):
        positions = np.array([i.pos for i in obj.atoms
                              if i.nodeid in chunk.keys()])
    elif isinstance(obj, BuildingUnit):
        positions = np.array([i.coordinates for i in obj.atoms
                              if i.nodeid in chunk.keys()])
    return np.sum(positions, axis=0) / float(len(positions))
    
def gen_distances(graph, obj):
    if isinstance(obj, FunctionalGroup):    
        coords = [atom.pos for atom in obj.atoms]
    elif isinstance(obj, BuildingUnit):
        coords = [atom.coordinates for atom in obj.atoms]
    distances = distance.cdist(coords, coords)
    atom_pairs = itertools.combinations(obj.atoms, 2)
    for pair in atom_pairs:
        nodeid1 = pair[0].nodeid
        nodeid2 = pair[1].nodeid
        nodexx1 = pair[0].idx
        nodexx2 = pair[1].idx
        dist = distances[nodexx1, nodexx2]
        graph[nodeid1]['distance'][nodeid2] = dist
        graph[nodeid2]['distance'][nodeid1] = dist


def gen_bu_graph(bu):
    _graph = {}
    for idx, atom in enumerate(bu.atoms):
        atom.idx = idx
        atom.nodeid = str(uuid4())
    for atom in bu.atoms:
        _graph.setdefault(atom.nodeid, {})
        _graph[atom.nodeid]['type'] = atom.force_field_type
        _graph[atom.nodeid]['element'] = atom.element
        _graph[atom.nodeid]['atomidx'] = atom.idx
        _graph[atom.nodeid]['distance'] = {}
        
        for bond in atom.bonds:
            atomid = bu.atoms[bond].nodeid
            _graph[atom.nodeid].setdefault('neighbours',[]).append(atomid)        
    return _graph
    
def N(v, g):
    try:
        return g[v]['neighbours']
    except KeyError:
        return []
    
def bk(R, P, X, g):
    """Bron-Kerbosch recursive algorithm"""
    if not any((P,X)):
        yield R
    for v in P[:]:
        R_v = R + [v]
        P_v = [v1 for v1 in P if v1 in N(v, g)] # p intersects N(vertex)
        X_v = [v1 for v1 in X if v1 in N(v, g)] # x intersects N(vertex)
        for r in bk(R_v, P_v, X_v, g):
            yield r
        P.remove(v)
        X.append(v)

def intersect(a, b):
    return list(set(a) & set(b))

def modulo(a, b):
    return [i for i in a if i not in b]

def pivot_choice(list, g):
    """Find the node with the highest connectivity."""
    max = [0,0]
    for node in list:
        n = N(node, g)
        if len(n) > max[1]:
            max = [node, len(n)]
    return max[0]

def bk_pivot(R, P, X, g):
    """Bron-Kerbosch recursive algorithm"""
    if not any((P,X)):
        yield R
    if intersect(P,X):
        union = intersect(P, X)
        u = pivot_choice(union)
        list = modulo(P[:], N(u, g))
    else:
        list = P[:]
    for v in list:
        R_v = R + [v]
        P_v = [v1 for v1 in P if v1 in N(v, g)] # p intersects N(vertex)
        X_v = [v1 for v1 in X if v1 in N(v, g)] # x intersects N(vertex)
        for r in bk_pivot(R_v, P_v, X_v, g):
            yield r
        P.remove(v)
        X.append(v)

def gen_correspondence_graph(g1, g2):
    """Generates a correspondence between graphs g1 and g2.
    Note, in the correspondence graph the first node listed
    in the pair is from g1 and the second is from g2.
    
    """
    # include atom typing for the equivalency.
    nodes_g1 = g1.keys()
    nodes_g2 = g2.keys()
    node_pairs = itertools.product(nodes_g1,nodes_g2)
    node_pairs = [pair for pair in node_pairs if isequal(pair, g1, g2)]
    _graph = {}
    for pair in node_pairs:
        _graph[tuple(pair)] = {}
        _graph[tuple(pair)]['neighbours'] = []
    return _graph
    # connect two nodes of the correspondence graph if their sub nodes are
    # connected

def gen_correspondence_neighbours(graph, g1, g2, tol=0.3):
    """Establishes 'edges' between nodes in the correspondence graph.
    Bond tolerance set to 0.1 angstroms... probably needs to
    be a bit more than this.
    
    """
    pairs = [pair for pair in itertools.combinations(graph, 2)
             if pair[0][0] != pair[1][0] and pair[0][1] != pair[1][1]]
    for pair in pairs:
        # determine if they are neighbours by their respective distances
        graph1_node1 = pair[0][0]
        graph1_node2 = pair[1][0]
        graph2_node1 = pair[0][1]
        graph2_node2 = pair[1][1]
        g1_dist = g1[graph1_node1]['distance'][graph1_node2]
        g2_dist = g2[graph2_node1]['distance'][graph2_node2]
        if abs(g1_dist - g2_dist) <= tol:
            # they are neighbours!
            graph[pair[0]]['neighbours'].append(pair[1])
            graph[pair[1]]['neighbours'].append(pair[0])

def isequal(node_pair, g1, g2):
    """takes a nested list and determines if the two nodes are
    'equivalent'.  The equivalency test shall be described later.
    
    """
    nodeid1 = node_pair[0]
    nodeid2 = node_pair[1]
    #if not (g1[nodeid1]['type'] == g2[nodeid2]['type']):
    if not (g1[nodeid1]['element'] == g2[nodeid2]['element']):
        return False
    return True
    # atom type checks of neighbours?
    # if not len(N(nodeid1, g1)) == len(N(nodeid2, g2)):
    #    return False
    #FIXME(pboyd): There is a different neighbour list for equivalent
    # atoms connected to 'connect points' in the original building
    # unit and to another 'bu' in the actual cif file.
    # need to introduce a 'wild card' for these atoms.
    #neighbour_types_g1 = [g1[i]['type'] for i in N(nodeid1, g1)]
    #neighbour_types_g2 = [g1[i]['type'] for i in N(nodeid1, g1)]
    #return sorted(neighbour_types_g1) == sorted(neighbour_types_g2)


def parse_mof_name(mofname):
    mof_ = mofname.split("_")
    mof__ = mof_[5].split(".")
    met = mof_[1]
    org1 = mof_[2]
    org2 = mof_[3]
    top = mof__[0]
    fnum = mof__[2]
    return met, org1, org2, top, fnum

def count_non_hydrogens(graph):
    return len([i for i, val in graph.items() if val['element'] != 'H'])
    
def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

def zero_cross(vector1, vector2):
    vector1 = vector1/np.linalg.norm(vector1)
    vector2 = vector2/np.linalg.norm(vector2)
    return np.allclose(np.zeros(3), np.cross(vector1, vector2), atol=0.01)
    
def match(clique, g1, g2, H_MATCH, SO3H_MATCH):
    """Match returned if the length is the same and the types
    are the same.
    clique is the nodes which are matched to g2, g1 is the
    graph the clique nodes belong to.
    """
    if H_MATCH and SO3H_MATCH:
        clique_types = [g1[i]['element'] for i in clique]
        node_types = [val['element'] for val in g2.values()]
    elif H_MATCH and not SO3H_MATCH:
        clique_types = [g1[i]['type'] for i in clique]
        node_types = [val['type'] for val in g2.values()]
    else:
        clique_types = [g1[i]['type'] for i in clique
                        if not g1[i]['type'].startswith("H")]
        node_types = [val['type'] for val in g2.values()
                      if not val['type'].startswith("H")]
    if not (len(clique_types) == len(node_types)):
        return False
    if not sorted(clique_types) == sorted(node_types):
        return False
    return True

def extract_clique(g1, g2, H_MATCH=True, SO3H_MATCH=False, tol=0.3):
    kk = {}
    C = gen_correspondence_graph(g1, g2)
    gen_correspondence_neighbours(C, g1, g2, tol=tol)
    #cliques = sorted(list(bk([], C.keys(), [], C)), reverse=True)
    for J in bk([], C.keys(), [], C):
        keys = [i[0] for i in J]
        if match(keys, g1, g2, H_MATCH, SO3H_MATCH):
            return {i: g1.pop(i) for i in keys}
    return {}
    #cliques = sorted(list(bk_pivot([], C.keys(), [], C)), reverse=True)
    # remove duplicate entries with just different order.
    #[kk.setdefault(tuple(sorted([j[0] for j in i])), 0) for i in cliques]
    #cliques = [i for i in kk.keys() if match(i, g1, g2, H_MATCH)]
    #if not cliques:
    #    return {}
    #return {i:g1.pop(i) for i in max(cliques, key=len)}

def gen_local_bus(mofname, bu_graphs):
    met, org1, org2, top, fnum = parse_mof_name(mofname)
    local_bus = []
    # found an error where the sra topology had two index 2's
    # o15 was also index 2... duh.
    if (met == 'm9') and (org1 == 'o2' or org2 == 'o2'):
        local_bus += ['o15']
    if (met == 'm2' or met == 'm3') and (top == 'pcu'):
        local_bus += [i for i in bu_graphs.keys() if org1 + "s" in i]
        local_bus += [i for i in bu_graphs.keys() if org2 + "s" in i]
    local_bus += [met, org1, org2]
    return {i: bu_graphs[i] for i in list(set((local_bus)))}


def add_name(nodeid, graph, name, _FGROUP=False, _NETID=False):
    if _FGROUP:
        graph[nodeid]['functional_group_name'] = name
    elif _NETID:
        graph[nodeid]['net_id'] = name
    else:
        graph[nodeid]['building_unit_name'] = name


def generate_bu_graphs():
    bu_graphs = {}
    bu_database = Database(options.building_unit_file)
    for bu in bu_database:
        prefix = "o" if not bu.metal else "m"
        bu_name = prefix + str(bu.index)
        bu_graphs[bu_name] = gen_bu_graph(bu)
        gen_distances(bu_graphs[bu_name], bu)
        for order, sbu in enumerate(bu.specialbu):
            sbu_name = bu_name + "s" + str(order)
            bu_graphs[sbu_name] = gen_bu_graph(sbu)
            gen_distances(bu_graphs[sbu_name], sbu)
    return bu_graphs

def generate_fnl_graphs():
    fnl_graphs = {}
    fnl_lib = FunctionalGroupLibrary()
    for name, obj in fnl_lib.items():
        if name != "H":
            fnl_graphs[name], junk = gen_graph_faps(obj)
            gen_distances(fnl_graphs[name], obj)
    return fnl_graphs

def clean(name):
    if name.startswith('./run_x'):
        name = name[10:]
    if name.endswith('.cif'):
        name = name[:-4]
    elif name.endswith('.niss'):
        name = name[:-5]
    elif name.endswith('.out-CO2.csv'):
        name = name[:-12]
    elif name.endswith('-CO2.csv'):
        name = name[:-8]
    elif name.endswith('.flog'):
        name = name[:-5]
    elif name.endswith('.out.cif'):
        name = name[:-8]
    elif name.endswith('.out'):
        name = name[:-4]
    elif name.endswith('.tar'):
        name = name[:-4]
    elif name.endswith('.db'):
        name = name[:-3]
    elif name.endswith('.faplog'):
        name = name[:-7]
    elif name.endswith('.db.bak'):
        name = name[:-7]
    elif name.endswith('.csv'):
        name = name[:-4]
    return name

def pop_chunks(cif_graph, chunk, net_chunks, ind):
    popind = []
    for i in chunk.keys():
        pind = net_chunks[ind].index(i)
        popind.append(pind)
        cif_graph.pop(i)
    for j in reversed(sorted(popind)):
        net_chunks[ind].pop(j)

def extract_fnl_chunks(mutable_cif_graph, underlying_net,
                       local_fnl_graphs, cif, net_chunks, tol):
    fnl_list = list(reversed(sorted([
                (count_non_hydrogens(g),i) for i, g in 
                local_fnl_graphs.items()])))
    for nchunk_ind, nchunk in enumerate(net_chunks):
        sub_graph = {i:mutable_cif_graph[i] for i in nchunk}
        for count, fnl in fnl_list:
            fnl_graph = local_fnl_graphs[fnl]
            SO3H_MATCH = True if fnl == "SO3H" else False
            chunk = extract_clique(sub_graph, fnl_graph,
                                   H_MATCH=True, SO3H_MATCH=SO3H_MATCH,
                                   tol=tol)
            while chunk:
                pop_chunks(mutable_cif_graph, chunk, net_chunks, nchunk_ind)
                net_id = add_net_node(underlying_net, chunk, cif, 
                                      _FNL=True, label=fnl)
                chunk = extract_clique(sub_graph, 
                            fnl_graph, H_MATCH=True, tol=tol)

def add_net_node(underlying_net, chunk, cif, 
                 _METAL=False, _ORGANIC=False, _FNL=False,
                 label=None):
    net_id = str(uuid4())
    com = calc_com(chunk, cif)
    underlying_net['nodes'][net_id] = {}
    underlying_net['nodes'][net_id]['atoms'] = chunk.copy()
    underlying_net['nodes'][net_id]['com'] = com
    underlying_net['nodes'][net_id]['label'] = label
    underlying_net['nodes'][net_id]['organic'] = _ORGANIC
    underlying_net['nodes'][net_id]['fgroup'] = _FNL
    underlying_net['nodes'][net_id]['metal'] = _METAL
    return net_id 

def extract_bu_chunks(mutable_cif_graph, underlying_net,
                      local_bu_graphs, cif, net_chunks, use_clique, tol):
    metal_list = ["Cu", "Zn", "Cd", "Cr", "V", "In", "Ba", "Zr", "Mn"]
    bu_list = list(reversed(sorted([
               (count_non_hydrogens(g), i) for i, g in
               local_bu_graphs.items()])))

    for count, bu in bu_list:
        bu_graph = local_bu_graphs[bu]
        _METAL, _ORGANIC = (bu.startswith('m'), bu.startswith('o'))
        for nchunk_ind, nchunk in enumerate(net_chunks):
            # this is ugly
            sub_graph = {i:mutable_cif_graph[i].copy() for i in nchunk}
            # quick and dirty fix for pillars
            if bu.startswith('m'):
                if any([i['element'] in metal_list for i in sub_graph.values()]):
                    pop_chunks(mutable_cif_graph, sub_graph, net_chunks, 
                               nchunk_ind)
                    net_id = add_net_node(underlying_net, sub_graph, 
                                          cif, _METAL=_METAL,
                                          _ORGANIC=_ORGANIC, label=bu)
            #elif count > 19 or bu == 'o9':
            if use_clique and not bu.startswith('m'):
                chunk = extract_clique(sub_graph, bu_graph, 
                               H_MATCH=False, tol=tol)
                while chunk:
                    pop_chunks(mutable_cif_graph, chunk, 
                               net_chunks, nchunk_ind)
                    net_id = add_net_node(underlying_net, chunk, 
                                          cif, _METAL=_METAL,
                                          _ORGANIC=_ORGANIC, label=bu)
                    chunk = extract_clique(sub_graph, bu_graph, 
                                    H_MATCH=False, tol=tol)
            elif not use_clique and not bu.startswith('m'):
                #c = count_non_hydrogens(sub_graph)
                net_atoms = sorted([i['element'] for i in sub_graph.values()
                                      if i['element'] != "H"])
                bu_atoms = sorted([i['element'] for i in bu_graph.values()
                                   if i['element'] != "H"])
                if net_atoms == bu_atoms:
                    pop_chunks(mutable_cif_graph, sub_graph, net_chunks, 
                               nchunk_ind)
                    net_id = add_net_node(underlying_net, sub_graph, 
                                          cif, _METAL=_METAL,
                                          _ORGANIC=_ORGANIC, label=bu)

def bonded_node(gr1, gr2):
    bonding_nodes1 = {node1: node2 for node1 in gr1.keys() for node2 in
                    gr2.keys() if node1 in gr2[node2]['neighbours']}
    bonding_nodes2 = {node2: node1 for node2 in gr2.keys() for node1 in
                      gr1.keys() if node2 in gr1[node1]['neighbours']}
    if sorted(bonding_nodes2.values()) != sorted(bonding_nodes1.keys()):
        warning("graph reports conflicting bonding")
    return [(node1, node2) for node1, node2 in bonding_nodes1.items()]

def proj_vu(v, u):
    """Projects the vector u onto the vector v."""
    return v*np.dot(u, v) / np.linalg.norm(v)

def vect_len(vector):
    return np.linalg.norm(vector)

def eval_edge(com1, atom1, com2, atom2, cif):
    """Returns an edge which is correctly oriented
    and shifted by the periodic boundaries.

    """
    edge = com1 - com2
    cell_orient = (0., 0., 0.)
    sp1 = get_atom(atom1, cif).scaled_pos
    sp2 = get_atom(atom2, cif).scaled_pos
    atom_bond = sp1 - sp2 
    reduced_atom_bond = atom_bond - np.round(atom_bond)
    # project onto the edge
    proj_e = proj_vu(edge, reduced_atom_bond)
    if np.allclose(
            np.dot(edge/vect_len(edge), proj_e/vect_len(proj_e)),
            -1.) or vect_len(atom_bond) > 0.5:
        # determine which cell direction to choose
        newcom = com2.copy()
        # iterate over all the 26 supercells and determine
        # the smallest vector distance.
        supercell = [i for i in itertools.product([1, -1, 0], repeat=3)
                     if not all([j==0 for j in i])]
        min = [0,99999999.]
        for p, q in enumerate(supercell):
            l = vect_len(com1 - newcom - q)
            if l < min[1]:
                cell_orient = supercell[p]
                edge = com1- newcom - q 
                min = [p, l]
    return edge, cell_orient 

def calc_edges(net, cif):
    """Calculates the edges between nodes of the net.
    This takes into account periodic boundaries, and
    includes the vectors and lengths of the vectors of
    each edge.

    Special case: m8, m9, m10 has self-bonding

    """
    edges = []
    k = [i for i in net['nodes'].keys()]
    net_pairs = list(itertools.combinations(k, 2))
    for n1, n2 in net_pairs:
        # get the atomistic nodes
        gr1 = net['nodes'][n1]['atoms']
        com1 = net['nodes'][n1]['com']
        gr2 = net['nodes'][n2]['atoms']
        com2 = net['nodes'][n2]['com']
        # re-calculated COMs
        common = bonded_node(gr1, gr2)
        for i1, i2 in common:
            at1 = get_atom(i1, cif)
            at2 = get_atom(i2, cif)
            # determine the vector which connects the two atoms
            edge_vector, cell_orientation = eval_edge(com1, i1, com2, i2, cif)
            edges.append((n1, n2, edge_vector, cell_orientation))
    return edges

def collect_remaining(net, g):
    """It seems that hydrogens get 'lost' in this algorithm
    so we'll collect them at the end.

    """
    netkeys = [i for i in net['nodes'].keys()]
    while g:
        
        for node, values in g.items():
            if values['element'] != 'H':
                warning("%s was not found associated with any "%(str(values['atomidx'])+
                    values['element'])+ "building unit or functional group!")
                return False
            neighbours = values['neighbours']

            for net_node in netkeys:
                netvals = net['nodes'][net_node]
                if any([i in netvals['atoms'].keys() for i in neighbours]):
                    try:
                        net['nodes'][net_node]['atoms'][node] = g.pop(node)
                    except KeyError:
                        warning("bonded atom already taken!")
                        return False
    return True 

def obtain_coordinating_nodes(node, jnode, net):
    """searches for common coordinating types, carboxylate,
    phosphonate ester, nitrogen heterocycle.

    """
    return_graph = {}
    graph = net['nodes'][jnode]['atoms'].copy()
    return_graph.update({node:graph[node]})
    neighbours = [i for i in N(node, graph) if i in graph.keys()]
    neighbour_types = [graph[i]['element'] for i in neighbours]
    if graph[node]['element'] == "C":
        if ["O", "O"] == sorted(neighbour_types):
            o_nodes = [i for i in neighbours if graph[i]['element'] == "O"]
            return_graph.update({i:graph[i] for i in o_nodes})

        elif ["C", "C"] == sorted(neighbour_types):
            extended_neighbours = [j for i in neighbours for j in 
                                   N(i, graph) if j != node]
            if ["H", "H", "N", "N"] == sorted([graph[i]['element'] 
                                        for i in extended_neighbours]):
                return_graph.update({i:graph[i] for i in 
                                     neighbours + extended_neighbours})

    elif graph[node]['element'] == "P":
        if ["O", "O", "O"] == sorted(neighbour_types):
            # problem here where one of the neighbours in N(i, graph) was
            # not one of the nodes in graph.
            neighbour_nodes = [j for i in neighbours for j in N(i, graph)
                               if j != node and j in graph.keys()]
            extended_neighbours = [j for j in neighbour_nodes 
                                   if graph[j]['element'] == 'C']
             
            c1_neighbours = [j for i in extended_neighbours for j in N(i, graph) if 
                              graph[j]['element'] != 'O']
            c1_neighbour_nodes = [j for i in c1_neighbours for j in N(i, graph)
                                  if j in graph.keys() and j not in 
                                  extended_neighbours]
            c2_neighbours = [j for j in c1_neighbour_nodes if
                              graph[j]['element'] != 'C']
            # note, not sure if Daff functionalized the ester moiety of the Ba
            # pillar, if he did, this does not capture the entire molecule.
            return_graph.update({i:graph[i] for i in neighbours +
                                 extended_neighbours + 
                                 c1_neighbours +
                                 c2_neighbours})
            
    return return_graph

def get_metal_cluster(node, edges, cif, net):
    metal_graph = {}
    lcif = copy.deepcopy(cif)
    metal_graph.update(net['nodes'][node]['atoms'].copy())
    # remove nodes not bonded to the metal unit 
    for pnode, vals in metal_graph.items():
        n = vals['neighbours']
        neighbour_pop = []
        for ind, i in enumerate(n):
            if i not in metal_graph.keys():
                neighbour_pop.append(ind)
        for p in reversed(sorted(neighbour_pop)):
            n.pop(p)
    # adjust the node positions to make one big molecule
    node_recalc = []
    for nnode in iter_neighbours(metal_graph.keys()[0], metal_graph):
        if nnode not in node_recalc:
            node_recalc.append(nnode)
            vals = metal_graph[nnode]
            atom = get_atom(nnode, lcif)
            neighbours = N(nnode, metal_graph)
            for nodeid in neighbours:
                adj_atom = get_atom(nodeid, lcif)
                diff = adj_atom.scaled_pos - atom.scaled_pos
                adj_atom.scaled_pos -= np.round(diff)
    # append the coordinates to the graph nodes
    for fnode in metal_graph.keys():
        atom = get_atom(fnode, lcif)
        metal_graph[fnode]['pos'] = np.dot(atom.scaled_pos, cif.cell.cell)
    return metal_graph


def get_organic_cluster(node, edges, cif, net):
    organic_graph = {}
    lcif = copy.deepcopy(cif)
    local_graph = net['nodes'][node]['atoms']
    organic_graph.update(local_graph.copy())
    node_edges = [i for i in edges if node in i[0:2]]
    joining_nodes = [k for i in node_edges for k in i[0:2] if k != node]
    for jnode in joining_nodes:
        jgraph = net['nodes'][jnode]['atoms']
        if net['nodes'][jnode]['metal']:
            # get the joining atom/s
            joining_atoms = [i for i in jgraph.keys() for k in 
                             local_graph.values() if i in k['neighbours']]
            for join in joining_atoms:
                # extract carboxylate/phosphonate/nitrogen heteroxycle
                organic_graph.update(obtain_coordinating_nodes(join, 
                                     jnode, net))
        else:
            organic_graph.update(net['nodes'][jnode]['atoms'].copy())
    # remove nodes not bonded to the organic unit 
    for pnode, vals in organic_graph.items():
        n = vals['neighbours']
        neighbour_pop = []
        for ind, i in enumerate(n):
            if i not in organic_graph.keys():
                neighbour_pop.append(ind)
        for p in reversed(sorted(neighbour_pop)):
            n.pop(p)
    # adjust the node positions to make one big molecule
    node_recalc = []
    for nnode in iter_neighbours(organic_graph.keys()[0], organic_graph):
        if nnode not in node_recalc:
            node_recalc.append(nnode)
            vals = organic_graph[nnode]
            atom = get_atom(nnode, lcif)
            neighbours = N(nnode, organic_graph)
            for nodeid in neighbours:
                adj_atom = get_atom(nodeid, lcif)
                diff = adj_atom.scaled_pos - atom.scaled_pos
                adj_atom.scaled_pos -= np.round(diff)
    # append the coordinates to the graph nodes
    for fnode in organic_graph.keys():
        atom = get_atom(fnode, lcif)
        organic_graph[fnode]['pos'] = np.dot(atom.scaled_pos, cif.cell.cell)
    return organic_graph

def gen_connect_edge_matrix(net, edge_space):

    order_dic = {}
    node_info = []
    edge_info = []
    net_n = [i for i in net['nodes'].keys()] 
    connect_matrix = np.zeros([len(net_n), len(net_n)])
    edge_matrix = np.zeros([len(edge_space), len(net_n)])
    for order, netnode in enumerate(net_n):
        order_dic[netnode] = order
        node_info.append({'com':net['nodes'][netnode]['com'],
                      'label':net['nodes'][netnode]['label'],
                      'organic':net['nodes'][netnode]['organic'],
                      'metal':net['nodes'][netnode]['metal'],
                      'fgroup':net['nodes'][netnode]['fgroup']
                      })
        if net['nodes'][netnode]['organic']:
            node_info[-1]['inchikey'] = net['nodes'][netnode]['inchikey']
        else:
            node_info[-1]['inchikey'] = None 

    for ind, (node1, node2, vector, orient) in enumerate(edge_space):
        o1, o2 = order_dic[node1], order_dic[node2]
        connect_matrix[o1, o2] = 1.
        connect_matrix[o2, o1] = 1.
        edge_matrix[ind, o1] = 1.
        edge_matrix[ind, o2] = 1.
        edge_info.append({'vector':vector, 'orientation':orient})
    return connect_matrix, edge_matrix, node_info, edge_info

def add_labels(net, edge_space, cif):
    """Add the building unit label to the attached functional
    groups.

    """
    count_dic = {}
    net_n = [i for i in net['nodes'].keys()] 
    for netnode in net_n:
        vals = net['nodes'][netnode]
        atoms = vals['atoms']
        label = vals['label']
        count_dic.setdefault(label, 0)
        count_dic[label] += 1
        vals['order'] = count_dic[label]
        if vals['metal'] or vals['organic']:
            for natom in atoms.keys():
                atoms[natom]['building unit'] = label
                atom = get_atom(natom, cif)
                atom.bu = label
                atom.molorder = count_dic[label]
                atom.fnl = '?'
            if vals['organic']:
                net['nodes'][netnode]['functionalization'] = {}
        elif vals['fgroup']:
            for natom in atoms.keys():
                atom = get_atom(natom, cif)
                atom.fnlorder = count_dic[label]
                atom.fnl = label
    for node1, node2, vector, orient in edge_space:
        if net['nodes'][node1]['organic'] and net['nodes'][node2]['fgroup']:
            atoms = net['nodes'][node2]['atoms']
            flabel = net['nodes'][node2]['label']
            olabel = net['nodes'][node1]['label']
            for natom in atoms.keys():
                atom = get_atom(natom, cif)
                atom.bu = olabel
                atom.molorder = net['nodes'][node1]['order'] 
                atoms[natom]['building unit'] = olabel
                atoms[natom]['functional group'] = flabel
            net['nodes'][node1]['functionalization'].setdefault(flabel, 0)
            net['nodes'][node1]['functionalization'][flabel] += 1

        elif net['nodes'][node1]['fgroup'] and net['nodes'][node2]['organic']:
            atoms = net['nodes'][node1]['atoms']
            flabel = net['nodes'][node1]['label']
            olabel = net['nodes'][node2]['label']
            for natom in atoms.keys():
                atom = get_atom(natom, cif)
                atom.bu = olabel
                atom.molorder = net['nodes'][node2]['order'] 
                atoms[natom]['building unit'] = olabel
                atoms[natom]['functional group'] = flabel
            net['nodes'][node2]['functionalization'].setdefault(flabel, 0)
            net['nodes'][node2]['functionalization'][flabel] += 1

def get_smiles(string):
    conv = pybel.ob.OBConversion()
    conv.SetOutFormat("smi")
    mol = pybel.readstring("mol", string)
    output = conv.WriteString(mol.OBMol)
    return output
def get_inchi(string):
    conv = pybel.ob.OBConversion()
    conv.SetOutFormat("inchi")
    mol = pybel.readstring("mol", string)
    output = conv.WriteString(mol.OBMol)
    return output
def get_inchikey(string):
    conv = pybel.ob.OBConversion()
    conv.SetOutFormat("inchikey")
    mol = pybel.readstring("mol", string)
    output = conv.WriteString(mol.OBMol)
    return output

def mol_string(graph, bond):
    header = "Organic\n Pete\n\n"
    counts = "%3i%3i%3i%3i%3i%3i%3i%3i%3i 0999 V2000\n"
    atom_block = "%10.4f%10.4f%10.4f %3s%2i%3i%3i%3i%3i%3i%3i%3i%3i%3i%3i%3i\n"
    bond_block = "%3i%3i%3i%3i%3i%3i%3i\n"

    mol_string = header
    mol_string += counts%(len(graph.keys()), len(bond.keys()), 0, 0, 0, 0, 0, 0, 0)
    atom_order = []
    for node, val in graph.items():
        atom_order.append(node)
        pos = val['pos']
        mol_string += atom_block%(pos[0], pos[1], pos[2], val['element'],
                                  0,0,0,0,0,0,0,0,0,0,0,0)
    for bond, type in bond.items():
        ind1 = atom_order.index(bond[0]) + 1
        ind2 = atom_order.index(bond[1]) + 1
        border = 4 if type == 1.5 else type
        mol_string += bond_block%(ind1, ind2, border, 0, 0, 0, 0)
    return mol_string

def plot_net(net):
    gp = GraphPlot()
    gp.plot_cell(cell=net['cell'], colour='g')
    nn = [i for i in net['nodes'].keys()]
    for node in nn:
        val = net['nodes'][node]
        if val['fgroup']:
            colour = 'b'
        elif val['organic']:
            colour = 'g'
        elif val['metal']:
            colour = 'r'
        gp.add_point(point=val['com'], label=val['label'], colour=colour)

    edge_space = net['edges']
    for edge in edge_space:
        com = net['nodes'][edge[0]]['com']
        gp.add_edge(edge[2], origin=com)
    gp.plot()

def mpi_chunks(seq, size):
    splitsize = 1.0/size*len(seq)
    for i in range(size):
        yield seq[int(round(i*splitsize)):int(round((i+1)*splitsize))]

def percent_complete(number, length):
    return float(number+1)/float(length) * 100.

def reduce_size(net):
    """Pop's out the distance key from the atoms"""
    for node, val in net['nodes'].items():
        #net['nodes'][node].pop('atoms')
        for atom in val['atoms'].keys():
            val['atoms'][atom].pop('distance')

def main():
    pybel.ob.obErrorLog.StopLogging()
    nets = {}
    mnets = {}
    inchikey_dic = {}
    inchi_dic = {}
    smiles_dic = {}
    rerun = []
    basename = options.csv_file[:-4]
    dummy = "dummy"
    # parallelize this stuff
    mofs = CSV(options.csv_file)
    rank = 0
    size = 1
    if rank == 0:
        data = list(mpi_chunks(mofs.keys(), size))
    else:
        data = None
    #data = comm.scatter(data, root=0)
    if not data:
        error("No MOFs to evaluate!")
    #    comm.Disconnect()
        sys.exit()
    bu_graphs = generate_bu_graphs()
    fnl_graphs = generate_fnl_graphs()
    functional_groups = FunctionalGroups(options.sql_file)
    for count, mof in enumerate(data[rank]):
        underlying_net = {'nodes':{}}
        info("rank %i, analyzing %s"%(rank, mof))
        cif = Structure(mof)
        try:
            cif.from_file(os.path.join(options.lookup, 
                         mof), "cif", dummy)
        except IOError:
            cif.from_file(os.path.join(options.lookup,
                          mof+".out"), "cif", dummy)
        underlying_net['cell'] = cif.cell.cell.copy()
        done = False
        iter_count = 0
        tol = options.tolerance
        store = True
        met, org1, org2, top, fnum = parse_mof_name(mof)
        if org1 != org2:
            use_clique = count_non_hydrogens(bu_graphs[org1]) == \
                         count_non_hydrogens(bu_graphs[org2])
        else:
            use_clique = False
        while not done:
            iter_count += 1
            cif_graph, cif_bonds = gen_graph_faps(cif)
            cut_graph, net_chunks = cut_mof_by_links(cif, cif_graph, mof)
            gen_cif_distance(cif, cut_graph, cif_graph)
            mutable_cif_graph = cif_graph.copy()
            local_fnl_grps = [i for i in functional_groups[mof].keys() if i 
                          and i != "H"]
            local_fnl_graphs = {i:k for i, k in fnl_graphs.items() if
                            i in local_fnl_grps}
            extract_fnl_chunks(mutable_cif_graph, underlying_net,
                           local_fnl_graphs, cif, net_chunks,
                           tol)
            local_bu_graphs = gen_local_bus(mof, bu_graphs)
            extract_bu_chunks(mutable_cif_graph, underlying_net,
                           local_bu_graphs, cif, net_chunks, use_clique,
                           tol)

            edge_space = calc_edges(underlying_net, cif)
            underlying_net['edges'] = edge_space
            if not collect_remaining(underlying_net, mutable_cif_graph):
                tol = tol*1.1
                underlying_net['nodes'] = {}
                underlying_net.pop('edges')
            else:
                done = True
            if iter_count >= 15:
                error("Problem finding cliques for %s"%mof)
                #skip to the next MOF
                store = False
                done = True
                #sys.exit()
        if store:
            add_labels(underlying_net, edge_space, cif)
            for node in [i for i in underlying_net['nodes'].keys()]:
                if underlying_net['nodes'][node]['organic']:
                    cluster = get_organic_cluster(node, edge_space, 
                                              cif, underlying_net)
                    local_bonds = {i:cif_bonds[i] for i in cif_bonds.keys() if 
                                 all([j in cluster.keys() for j in i])}
                    mol = mol_string(cluster, local_bonds)
                    #molfile = open(underlying_net['nodes'][node]['label']+".mol","w")
                    #molfile.writelines(mol)
                    #molfile.close()
                    smiles = get_smiles(mol).strip()
                    inchi = get_inchi(mol).strip()
                    inchikey = get_inchikey(mol).strip()
                    underlying_net['nodes'][node]['smiles'] = smiles
                    underlying_net['nodes'][node]['inchi'] = inchi
                    underlying_net['nodes'][node]['inchikey'] = inchikey
                    #underlying_net['nodes'][node]['mol'] = mol
                    inchikey_dic.setdefault(inchikey, []).append((node, mof))
                    inchi_dic.setdefault(inchi, []).append((node, mof))
                    smiles_dic.setdefault(smiles, []).append((node, mof))
                elif underlying_net['nodes'][node]['metal']:
                    cluster = get_metal_cluster(node, edge_space,
                                                cif, underlying_net)
                    local_bonds = {i:cif_bonds[i] for i in cif_bonds.keys() if
                                   all([j in cluster.keys() for j in i])}
                    mol = mol_string(cluster, local_bonds)
                    #molfile = open(underlying_net['nodes'][node]['label']+".mol","w")
                    #molfile.writelines(mol)
                    #molfile.close()
                    smiles = get_smiles(mol).strip()
                    inchi = get_inchi(mol).strip()
                    inchikey = get_inchikey(mol).strip()
                    underlying_net['nodes'][node]['smiles'] = smiles
                    underlying_net['nodes'][node]['inchi'] = inchi
                    underlying_net['nodes'][node]['inchikey'] = inchikey
                    #underlying_net['nodes'][node]['mol'] = mol
                    inchikey_dic.setdefault(inchikey, []).append((node, mof))
                    inchi_dic.setdefault(inchi, []).append((node, mof))
                    smiles_dic.setdefault(smiles, []).append((node, mof))
                    
            lines = to_cif(cif.atoms, cif.cell, cif.bonds, mof + ".lab")
            cif_name = mof + ".lab.cif" 
            output_file = open(cif_name, "w")
            output_file.writelines(lines)
            output_file.close()
            reduce_size(underlying_net)
            #pickle_file = open('%s.pkl'%mof, 'wb')
            #pickle.dump(underlying_net, pickle_file)
            #pickle_file.close()
            nets[mof] = underlying_net
            connect, edge, c_info, e_info = \
                    gen_connect_edge_matrix(underlying_net, edge_space)
    
            mnets[mof] = {'cell':underlying_net['cell'],
                          'connectivity_matrix':connect,
                          'edge_matrix':edge,
                          'node_info':c_info,
                          'edge_info':e_info}

        else:
            rerun.append(mof)

        if (count % options.report_frequency == 0 or count == (len(data[rank])-1)):
            p = percent_complete(count, len(data[rank]))
            info("rank %i is %5.3f %% complete"%(rank, p))
            
    #node_list =  comm.gather(nets, root=0)
    #mnode_list = comm.gather(mnets, root=0)
    if rank == 0:
        #for i in node_list:
        #    nets.update(i)
        #for i in mnode_list:
        #    mnets.update(i)
        pickle_file = open('m_%s.pkl'%(basename), 'wb')
        pickle.dump(mnets, pickle_file)
        pickle_file.close()
        pickle_file = open("net_%s.pkl"%(basename), "wb")
        pickle.dump(nets, pickle_file)
        pickle_file.close()
        info("MOFs to re-analyze")
        for m in rerun:
            info("%s"%(m))
    #plot_net(underlying_net)
if __name__ == "__main__":
    main()
