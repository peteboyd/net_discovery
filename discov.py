#!/usr/bin/env python

"""database.py

This program was designed to scan through mass amounts of data to compile what
is deemed necessary information for each MOF.  This will include building units
where the functional group is located, uptake at given conditions, and other
such stuff.

"""
import itertools
import pickle
import math
from uuid import uuid4
from faps.faps import Structure, Cell, Atom, Symmetry
from genstruct.genstruct import Database, BuildingUnit, Atom_gen
from faps.function_switch import FunctionalGroupLibrary, FunctionalGroup
from scipy.spatial import distance
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
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

class GraphPlot(object):
    
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        
    def plot_cell(self, cell=np.identity(3), origin=np.zeros(3), colour='b'):
        # add axes labels
        xyz_a = (cell[0]+origin)/2.
        xyz_b = (cell[1]+origin)/2.
        xyz_c = (cell[2]+origin)/2.
        self.ax.text(xyz_a[0], xyz_a[1], xyz_a[2], 'a')
        self.ax.text(xyz_b[0], xyz_b[1], xyz_b[2], 'b')
        self.ax.text(xyz_c[0], xyz_c[1], xyz_c[2], 'c')

        all_points = [np.sum(a, axis=0)+origin
                      for a in list(powerset(cell)) if a]
        all_points.append(origin)
        for s, e in itertools.combinations(np.array(all_points), 2):
            if any([zero_cross(s-e, i) for i in cell]):
                self.ax.plot3D(*zip(s, e), color=colour)
                
    def add_point(self, point=np.zeros(3), label=None, colour='r'):
        
        self.ax.scatter(*point, color=colour)
        if label:
            self.ax.text(*point, s=label)
            
    def plot(self):
        plt.show()

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
        
def round_i(i):
    """Sorts the fractional value i to the minimum image"""
    if i > 0.5:
        i = i - 1
    elif i < -0.5:
        i = i + 1
    return i

def gen_cif_distance(newcif, graph):
    inv = newcif.cell.inverse
    for atom in newcif.atoms:
        atom.scaled_pos = np.array(atom.fpos(inv))
    #print len(newcif.atoms)
    atom_pairs = list(itertools.combinations(newcif.atoms, 2))
    for pair in atom_pairs:
        dist = pair[0].scaled_pos - pair[1].scaled_pos
        v_dist = np.array([round_i(i) for i in dist])
        vect = np.dot(v_dist, newcif.cell.cell)
        length = np.linalg.norm(vect)
        graph[pair[0].nodeid]['distance'][pair[1].nodeid] = length
        graph[pair[1].nodeid]['distance'][pair[0].nodeid] = length
        
def gen_graph_faps(faps_obj):
    _graph = {}
    for idx, atom in enumerate(faps_obj.atoms):
        atom.idx = idx
        atom.nodeid = str(uuid4())
        _graph.setdefault(atom.nodeid, {})
        _graph[atom.nodeid]['neighbours'] = []
        _graph[atom.nodeid]['atomidx'] = idx
        _graph[atom.nodeid]['type'] = atom.uff_type
        _graph[atom.nodeid]['distance'] = {}

    for bond in faps_obj.bonds:
        atomid1 = faps_obj.atoms[bond[0]].nodeid
        atomid2 = faps_obj.atoms[bond[1]].nodeid
        _graph.setdefault(atomid1, {})
        _graph.setdefault(atomid2, {})
        _graph[atomid1]['neighbours'].append(atomid2)
        _graph[atomid2]['neighbours'].append(atomid1)
    return _graph

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
            neighbours = vals['neighbours']
            for nodeid in neighbours:
                adj_atom = get_atom(nodeid, obj)
                diff = adj_atom.scaled_pos - atom.scaled_pos
                adj_atom.scaled_pos -= np.round(diff)
        # ignore the hydrogen atoms for the com calc
        positions = np.array([i.scaled_pos for i in obj.atoms
                              if i.nodeid in chunk.keys() and
                              not i.uff_type.startswith("H")])                
        com = np.array([i - math.floor(i) for i in
                        np.sum(positions, axis=0) / float(len(positions))])
        return np.dot(com, obj.cell.cell)
    
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

def gen_correspondence_neighbours(graph, g1, g2, tol = 0.1):
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
        if g1_dist - g2_dist <= tol:
            # they are neighbours!
            graph[pair[0]]['neighbours'].append(pair[1])
            graph[pair[1]]['neighbours'].append(pair[0])

def isequal(node_pair, g1, g2):
    """takes a nested list and determines if the two nodes are
    'equivalent'.  The equivalency test shall be described later.
    
    """
    nodeid1 = node_pair[0]
    nodeid2 = node_pair[1]
    if not (g1[nodeid1]['type'] == g2[nodeid2]['type']):
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
    heavycount = 0
    for node, keys in graph.items():
        if not keys['type'].startswith("H"):
            heavycount += 1
    return heavycount
    
def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

def zero_cross(vector1, vector2):
    vector1 = vector1/np.linalg.norm(vector1)
    vector2 = vector2/np.linalg.norm(vector2)
    return np.allclose(np.zeros(3), np.cross(vector1, vector2), atol=0.01)
    
def match(clique, g1, g2, H_MATCH):
    """Match returned if the length is the same and the types
    are the same.
    clique is the nodes which are matched to g2, g1 is the
    graph the clique nodes belong to.
    """
    if H_MATCH:
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

def extract_clique(g1, g2, newcif, H_MATCH=True):
    kk = {}
    C = gen_correspondence_graph(g1, g2)
    gen_correspondence_neighbours(C, g1, g2, tol=0.3)
    cliques = sorted(list(bk([], C.keys(), [], C)), reverse=True)
    # remove duplicate entries with just different order.
    [kk.setdefault(tuple(sorted([j[0] for j in i])), 0) for i in cliques]
    cliques = [i for i in kk.keys() if match(i, g1, g2, H_MATCH)]
    if not cliques:
        return {}
    return {i:g1.pop(i) for i in max(cliques, key=len)}

def gen_local_bus(mofname, bu_graphs):
    met, org1, org2, top, fnum = parse_mof_name(mofname)
    local_bus = []
    if (met == 'm2' or met == 'm3') and (top == 'pcu'):
        local_bus += [i for i in bu_graphs.keys() if org1 + "s" in i]
        local_bus += [i for i in bu_graphs.keys() if org2 + "s" in i]
    local_bus += [met, org1, org2]
    return list(set((local_bus)))

def bonded_node(gr1, gr2):
    bonding_nodes1 = {node1: node2 for node1 in gr1.keys() for node2 in
                    gr2.keys() if node1 in gr2[node2]['neighbours']}
    bonding_nodes2 = {node2: node1 for node2 in gr2.keys() for node1 in
                      gr1.keys() if node2 in gr1[node1]['neighbours']}
    if sorted(bonding_nodes2.values()) != bonding_nodes1.keys():
        print "WARNING: graph reports conflicting bonding"
    return [(node1, node2) for node1, node2 in bonding_nodes1.items()]

def add_name(nodeid, graph, name, _FGROUP=False, _NETID=False):
    if _FGROUP:
        graph[nodeid]['functional_group_name'] = name
    elif _NETID:
        graph[nodeid]['net_id'] = name
    else:
        graph[nodeid]['building_unit_name'] = name


def main():
    underlying_net = {}
    gp = GraphPlot()
    bu_graphs, fnl_graphs = {}, {}
    dummy = "dummy"
    structure_name = 'str_m2_o1_o11_f0_pcu.sym.42'
    newcif = Structure(structure_name)
    newcif.from_file("testdir/%s"%(structure_name), "cif", dummy)
    cif_graph = gen_graph_faps(newcif)
    gp.plot_cell(cell=newcif.cell.cell, colour='g')
    bu_database = Database("testdir/met2pcu.dat")
    for bu in bu_database:
        prefix = "o" if not bu.metal else "m"
        bu_name = prefix + str(bu.index)
        bu_graphs[bu_name] = gen_bu_graph(bu)
        gen_distances(bu_graphs[bu_name], bu)
        for order, sbu in enumerate(bu.specialbu):
            sbu_name = bu_name + "s" + str(order)
            bu_graphs[sbu_name] = gen_bu_graph(sbu)
            gen_distances(bu_graphs[sbu_name], sbu)
    
    local_bus = gen_local_bus(structure_name, bu_graphs)
    # sort in decreasing order of bu length.
    gen_cif_distance(newcif, cif_graph)
    mutable_cif_graph = cif_graph.copy()
    #===================
    # Functional Groups
    #===================
    fnl_lib = FunctionalGroupLibrary()
    for name, obj in fnl_lib.items():
        fnl_graphs[name] = gen_graph_faps(obj)
        gen_distances(fnl_graphs[name], obj)

    f_test = fnl_graphs['NO2']
    chunk = extract_clique(mutable_cif_graph, f_test, newcif, H_MATCH=True)
    while chunk:
        com = calc_com(chunk, newcif)
        net_id = uuid4()
        underlying_net[net_id] = {}    
        underlying_net[net_id]['nodes'] = chunk
        underlying_net[net_id]['com'] = com
        underlying_net[net_id]['label'] = 'NO2'       
        gp.add_point(com, colour='g', label='NO2')
        chunk = extract_clique(mutable_cif_graph, f_test, newcif, H_MATCH=True)

    #===================
    # Building Units
    #===================
    local_bus.pop(local_bus.index('m2'))
    for bu in local_bus:
        print bu
        graph = bu_graphs[bu]
        n_heavy = count_non_hydrogens(graph)
        print "number of non-hydrogen atoms = %i"%(n_heavy)
        chunk = extract_clique(mutable_cif_graph, graph, newcif, H_MATCH=False)
        while chunk:
            com = calc_com(chunk, newcif)
            net_id = uuid4()
            underlying_net[net_id] = {}
            underlying_net[net_id]['nodes'] = chunk
            underlying_net[net_id]['com'] = com
            underlying_net[net_id]['label'] = bu
            gp.add_point(com, colour='r', label=bu)
            print "leftover atoms = %i"%(len(mutable_cif_graph.keys()))
            chunk = extract_clique(mutable_cif_graph, graph, newcif, H_MATCH=False)

    net_pairs = itertools.combinations(underlying_net.keys(), 2)
    for pair in net_pairs:

        
        graph1 = underlying_net[pair[0]]['nodes']
        graph2 = underlying_net[pair[1]]['nodes']
        bonds = bonded_node(graph1, graph2)
        if bonds:
            print underlying_net[pair[0]]['label'], underlying_net[pair[1]]['label']
            print bonds
    for netnode, stuff in underlying_net.items():
        print netnode
        
    gp.plot()
if __name__ == "__main__":
    main()
