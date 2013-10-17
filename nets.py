from correspondence_graphs import CorrGraph
from sub_graphs import SubGraph, OrganicSBU
from SecondaryBuildingUnit import SBU
import itertools
import ConfigParser
import sys
import os
from plotter import GraphPlot
import numpy as np
from copy import copy
from logging import debug, warning
from scipy.spatial import distance
from CIFer import CIF
from elements import CCDC_BOND_ORDERS
from element_properties import METALS
#from memory_profiler import profile
np.set_printoptions(threshold='nan')

class Net(object):
    # coordinating species
    species = {'m1':'carboxylate', 'm2':'carboxylate',
               'm3':'carboxylate', 'm4':'carboxylate',
               'm5':'carboxylate', 'm6':'tetrazole',
               'm7':'carboxylate', 'm8':'carboxylate',
               'm9':'carboxylate', 'm10':'phosphonateester',
               'm11':'carboxylate','m12':'pyrazole'}

    def __init__(self, options, mof):
        """Convert a faps Structure object into a net."""
        self.options = options
        self.mof = mof
        self.name = mof.name
        self.cell = mof.cell.cell
        self.icell = np.linalg.inv(mof.cell.cell.T)
        self.fragments = []
        self.nodes = []
        self.edge_vectors = None
        self.edge_matrix = None
        self.main_sub_graph = SubGraph(self.options, self.mof.name)
        self.cutted_bonds = {}
        self.fragmented_sub_graph = [] 
        if options.mofs_from_groin:
            self.main_sub_graph.from_faps(mof)
            self.main_sub_graph.compute_bonds(options.supercell*self.cell)

    def cut_sub_graph_by_coordination(self):
        """Cuts the bonds between coordinating groups and the SBUs.
        This allows a more memory-efficient way of finding the SBUs.
        """
        organic_dic = {}
        coord_units = self.coordination_units
        met, k, k, k, k = self.parse_groin_mofname()
        # special cases - cut by N-Metal bond.
        n_bond = ['m2', 'm3']
        clq = CorrGraph(self.options, copy(self.main_sub_graph))
        unit = coord_units[self.species[met]]
        clq.pair_graph = unit
        cliques = self.gen_cliques(clq, NO_H=False)
        for clique in cliques:
            #clique.debug('coordinator')
            # get atoms which are bonded to the clique
            atoms = self.main_sub_graph.get_bonding_atoms(clique._new_index)
            # identify the SBU cutting bond
            cut_bond = self.get_inter_sbu_bond(clique._new_index, atoms, met)
            # cut the bond
            if cut_bond is not None:
                self.cutted_bonds[cut_bond] = self.main_sub_graph.bonds.pop(cut_bond)

        if met in n_bond:
            self.cut_metal_nitrogen_bond()

    def cut_metal_nitrogen_bond(self):
        # find the N-coordinating ligands
        for atom in range(len(self.main_sub_graph)):
            if self.main_sub_graph[atom] in METALS:
                atoms = self.main_sub_graph.get_bonding_atoms([atom])
                for at in atoms:
                    if self.main_sub_graph[at] == "N":
                        cut_bond = tuple([at, atom])
                        try:
                            self.cutted_bonds[cut_bond] = self.main_sub_graph.bonds.pop(cut_bond)
                        except KeyError:
                            cut_bond = tuple([atom, at])
                            self.cutted_bonds[cut_bond] = self.main_sub_graph.bonds.pop(cut_bond)

    def get_inter_sbu_bond(self, atoms1, atoms2, met):
        """Return the two atoms which are bonded via an sbu bond."""
        elem = self.main_sub_graph.elements
        # elem 1 should always (?) be the non-metallic atom
        # elem 2 should always (?) be a carbon atom.
        for at1, at2 in itertools.product(atoms1, atoms2):
            if met == 'm10':
                coord_atom = "P"
            else:
                coord_atom = "C"
            if (at1, at2) in self.main_sub_graph.bonds.keys() or\
                    (at2, at1) in self.main_sub_graph.bonds.keys():
                
                if elem[at1] == coord_atom and elem[at2] == "C":
                    b = tuple([at1, at2])
                    try:
                        self.main_sub_graph.bonds[b]
                        return b
                    except KeyError:
                        return tuple([at2, at1])

    def fragmentate(self):
        done = False
        atoms = range(len(self.main_sub_graph))
        while not done:
            atid = atoms[0]
            sys.setrecursionlimit(10000)
            try:
                frag = self.main_sub_graph.fragmentizer(atid, r=[], q=[])
            except MaximumRecursionDepth:
                warning("Too high a recursion for the mof, returning..")
                return
            # PETE - CHECK IF THE VARIABLE FRAG IS DUPLICATED IN FRAGMENTED_SUB_GRAPH
            # THERE IS SOME REASON WHY BOTH O1 AND O4 ARE BEING FOUND IN THE SAME
            # FRAGMENT.
            sub_g = self.main_sub_graph % frag
            self.fragmented_sub_graph.append(sub_g)
            frag_ids = [atoms.index(i) for i in frag]
            for k in reversed(sorted(frag_ids)):
                atoms.pop(k)
            if not atoms:
                done = True
        # re-instate the cutted bonds
        self.main_sub_graph.bonds.update(self.cutted_bonds)

    def get_groin_sbus(self, sbus):
        met, o1, o2, top, fnl = self.parse_groin_mofname()
        sbu_list = []
        local_bus = [met, o1, o2]
        # fix for a mistake when generating structures.
        if (met == '9') and (o1 == '2' or o2 == '2'):
            local_bus += ['15']
        for i, sbu in sbus.items():
            if i.rstrip('s') in local_bus:
                sbu.name = i
                non_hydrogen_count = len([i for i in sbu.atoms 
                                      if i.element != "H"])
                sbu_list.append((non_hydrogen_count, sbu))
        return [i[1] for i in reversed(sorted(sbu_list))]

    def from_fragmentated_mof(self, sbus, fnls):
        """Extract from fragments"""
        sbu_list = self.get_groin_sbus(sbus)
        id = 0
        for frag in self.fragmented_sub_graph:
            self.extract_fragments(sbus, fnls, frag=frag)
            id+= 1

    def extract_fragments(self, sbus, fnls, frag=None):
        """Extract the sbus from groin mofs."""
        debug("Size of mof = %i"%(len(self.mof.atoms)))
        sbu_list = self.get_groin_sbus(sbus)
        if frag is None:
            sub_graph_copy = self.main_sub_graph % range(len(self.main_sub_graph))
            clq = CorrGraph(self.options, sub_graph_copy)
        else:
            sub_graph_copy = frag % range(len(frag))
            clq = CorrGraph(self.options, sub_graph_copy)
            
        if any(fnls):
            for fnl in fnls:
                clq.pair_graph = fnl
                generator = self.gen_cliques(clq, NO_H=False)
                for clique in generator:
                    #print float(sys.getsizeof(self.fragments))/ 1.049e6, " Mb"
                    self.fragments.append(clique)
                    #clique.debug("fnls")
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
                self.fragments.append(clique)
                #clique.debug("sbus")

        debug("There remains %i un-attached nodes. These will"%(len(clq.sub_graph)) +
                " be ignored as they are considered fragments outside "+
                "the periodic boundaries.")
        self.get_dangling_hydrogens(clq.sub_graph)
        del clq

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
            self.nodes.append((frag.name, frag.centre_of_atoms))

    def prune_unit_cell(self):
        """Keep only those nodes which reside in the unit cell.
        Remove edges outside the unit cell."""
        # This is disgusting code.
        images = []
        fractionals = []
        for id, (name, node) in enumerate(self.nodes):
            frac = np.dot(self.icell, node)
            fractionals.append(frac)
            if any(np.logical_or(frac>1., frac<-0.0001)):
                images.append(id)
        # find image in unit cell
        unit_cells = [i for i in range(len(self.nodes)) if i not in images] 
        correspondence = range(len(self.nodes))
        for node in images:
            f = fractionals[node]
            min_img = np.array([i%1 for i in f])
            # NOTE: THE ATOL set here seems extremely high for fractional
            # coordinate images. This may need to be adjusted
            node_img = [i for i in unit_cells if np.allclose(fractionals[i], 
                        min_img, atol=1e-3)]
            if len(node_img) == 0:
                # Just delete these nodes - they are likely cliques of fragmentated
                # SBUs at the periodic boundaries.
                warning("Could not find the image for one of the fragments!")
                warning("Type: %s"%self.fragments[node].name)
                warning("Fractional Corrdinates: %7.5f %7.5f %7.5f"%(tuple(f.tolist())))
            elif len(node_img) > 1:
                warning("Multiple images found in the unit cell!!!")
                return False
            else:
                image = node_img[0]
                correspondence[node] = image
        # finally, adjust the edge matrix to correspond to the minimum image
        edge_pop = []
        for edge_id, edge in enumerate(self.edge_matrix):
            ids = [id for id, i in enumerate(edge) if i]
            # if any node in an edge matrix row corresponds to image bonding
            # with an image, delete.
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
            # delete images from the edge matrix
            self.edge_matrix = np.delete(self.edge_matrix, xx, axis=1)
        for xy in reversed(sorted(edge_pop)):
            self.edge_matrix = np.delete(self.edge_matrix, xy, axis=0)
            self.edge_vectors.pop(xy)
        return True

    def edge_exists(self, sub1, sub2):
        return any([tuple(sorted(list(i))) in self.main_sub_graph.bonds.keys() for i
            in itertools.product(sub1._new_index, sub2._new_index)])

    def bond_exists(self, sub1, sub2):
        """Returns the indices of the atom elements found in sub1 and sub2
        in the form (sub1 index, sub2 index) if they are bonded together.
        """
        for i in itertools.product(sub1._orig_index, sub2._orig_index):
            if tuple(sorted(list(i))) in self.mof.bonds.keys():
                return (sub1._orig_index.index(i[0]),
                        sub2._orig_index.index(i[1]))
        return () 

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

        clq.correspondence_api()
        #clq.correspondence()
        try:
            mem = (clq.adj_matrix.nbytes + sys.getsizeof(clq.nodes)) / 1.049e6
        except AttributeError:
            mem = 0.0
            clq.adj_matrix = np.zeros(1)
            return 
        debug("Memory allocation for correspondence graph with %s"%(clq.pair_graph.name) +
              " requires %9.3f Mb"%(mem))
        mc = clq.extract_clique()
        remove = []
        #replace = []
        while not done:
            #clq.correspondence()
            if not clq.size:
                for xx in reversed(sorted(remove)):
                    del clq.sub_graph[xx]
                remove = []
                return

            sub_nodes = sorted(mc.next())
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
                # permanently remove these nodes from the sub_graph
                # so they are not found by other iterations.
                remove += sub_nodes[:]
                #for xx in reversed(sorted(sub_nodes)):
                #    del clq.sub_graph[xx]

                yield clique
            # in this instance we have found a clique not
            # belonging to the SBU. ignore
            elif len(all_elem) >= len(compare_elements):
                pass
                #replace.append(clq.sub_graph % sub_nodes)
                #for xx in reversed(sorted(sub_nodes)):
                #    del clq.sub_graph[xx]
            else:
                #for sub in replace:
                #    clq.sub_graph += sub
                for xx in reversed(sorted(remove)):
                    del clq.sub_graph[xx]
                #clq.sub_graph.debug()
                remove = []
                done = True


    def parse_groin_mofname(self):
        """metal, organic1, organic2, topology, functional group code"""
        ss = self.mof.name.split("_")
        met = ss[1]
        o1 = ss[2]
        o2 = ss[3]
        pp = ss[-1].split('.')
        top = pp[0]
        try:
            fnl = pp[2]
        except IndexError:
            fnl = '0'
        return met, o1, o2, top, fnl

    def show(self):
        gp = GraphPlot()
        gp.plot_cell(cell=self.cell, colour='g')
        for id, (name, node) in enumerate(self.nodes):
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

    @property
    def coordination_units(self):
        try:
            return self._coord_units
        except AttributeError:
            self._coord_units = {}
            for file in self.options.coord_unit_files:
                config = ConfigParser.SafeConfigParser()
                config.read(os.path.expanduser(file))
                for _io in config.sections():
                    coord = SBU()
                    coord.from_config(_io, config)
                    # create the subgraph
                    graph = SubGraph(self.options, _io)
                    graph.from_sbu(coord)
                    self._coord_units[_io] = graph
            return self._coord_units

    def organic_data(self):
        """Determine the organic SBUs in the list of fragments,
        then add the coordinating atoms, compute the inchikey,
        report.."""
        #TODO(pboyd): this probably could be a little more
        # 'object oriented'
        organic_dic = {}
        coord_units = self.coordination_units
        met, b,b,b,b = self.parse_groin_mofname()
        for id, (name, node) in enumerate(self.nodes):
            frag = self.fragments[id]
            if not frag.name.startswith('o'):
                continue
            # determine bonding nodes
            org_sbu = OrganicSBU(self.options, name=frag.name)
            org_sbu += frag 
            neighbours = set(self.get_neighbours(id))
            for neighbour in neighbours:
                n_frag = self.fragments[neighbour]
                # metal test
                if n_frag.name.startswith('m'):
                    graph = n_frag % range(len(n_frag))
                    clq = CorrGraph(self.options, graph)
                    # the following assumes a groin mof
                    clq.pair_graph = coord_units[self.species[met]]
                    generator = self.gen_cliques(clq)
                    for g in generator:
                        btest = self.bond_exists(frag, g)
                        if btest:
                            f1, f2 = btest
                            # do some shifting.
                            self.min_img_shift(frag._coordinates[f1],
                                                g)
                            # append g to the subgraph.
                            org_sbu += g
                    continue

                fnl_bond = self.bond_exists(frag, n_frag)
                if fnl_bond:
                    b1, b2 = fnl_bond
                    self.min_img_shift(frag._coordinates[b1],
                                       n_frag)
                    org_sbu += n_frag
            # just copy over all the bonds in the original MOF..
            # i don't care anymore.
            org_sbu.bonds = {}
            org_sbu.update_bonds(self.mof.bonds)
            #org_sbu.debug('org_sbu')
            organic_dic.update({org_sbu.inchikey(): org_sbu.to_mol()})
        return organic_dic

    def min_img_shift(self, coordinate, subg):
        """Shifts the subgraph 'subg's coordinates to the minimum
        image of coordinate"""
        fcoord = np.dot(self.icell, coordinate)
        for id, coord in enumerate(subg._coordinates):
            scaled = np.dot(self.icell, coord)
            shift = np.around(fcoord - scaled)
            subg._coordinates[id] = np.dot(scaled + shift, self.cell)

    def get_neighbours(self, idx):
        """Return the indices of the nodes corresponding to neighbours
        of the node who's entry is id."""
        neighbours = []
        for edge in self.edge_matrix:
            bonded_nodes = [id for id, i in enumerate(edge) if i]
            # in some cases fragmented nodes will create problems
            # with the unit cell bonding
            if not len(bonded_nodes) == 2:
                continue
            else:
                if idx in bonded_nodes:
                    try:
                        rem = bonded_nodes.index(idx)
                        add_node = bonded_nodes[1%rem]
                    except ZeroDivisionError:
                        add_node = bonded_nodes[1]
                    neighbours.append(add_node)
        return neighbours

    def to_cif(self):
        c = CIF(name=self.name+".net")
        # place the fragment info right before the atom block
        c.insert_block_order("fragment", 4)
        c.add_data("data", data_=self.name)
        c.add_data("data", _audit_creation_date=CIF.label(c.get_time()))
        c.add_data("data", _audit_creation_method=CIF.label("Net Finder v.%4.3f"%(
                                                            self.options.version)))
        # sym block
        c.add_data("sym", _symmetry_space_group_name_H_M=CIF.label("P1"))
        c.add_data("sym", _symmetry_Int_Tables_number=CIF.label("1"))
        c.add_data("sym", _symmetry_cell_setting=CIF.label("triclinic"))

        c.add_data("sym_loop", _symmetry_equiv_pos_as_xyz=
                                CIF.label("'x, y, z'"))
        c.add_data("cell", _cell_length_a=CIF.cell_length_a(self.mof.cell.a))
        c.add_data("cell", _cell_length_b=CIF.cell_length_b(self.mof.cell.b))
        c.add_data("cell", _cell_length_c=CIF.cell_length_c(self.mof.cell.c))
        c.add_data("cell", _cell_angle_alpha=CIF.cell_angle_alpha(self.mof.cell.alpha))
        c.add_data("cell", _cell_angle_beta=CIF.cell_angle_beta(self.mof.cell.beta))
        c.add_data("cell", _cell_angle_gamma=CIF.cell_angle_gamma(self.mof.cell.gamma))

        labels, indices = [], []
        fcoords = []
        for order, i in enumerate(self.fragments):
            c.add_data("fragment", _chemical_identifier=CIF.label(order),
                                   _chemical_name=CIF.label(i.name))
            for id in i._orig_index:
                # take original info from atom
                indices.append(id)
                atom = self.mof.atoms[id]
                element = atom.type
                label = c.get_element_label(element)
                labels.append(label)
                type = atom.uff_type
                pos = atom.ifpos(self.icell)
                fcoords.append(pos)
                c.add_data("atoms", _atom_site_label=CIF.atom_site_label(label))
                c.add_data("atoms", _atom_site_type_symbol=CIF.atom_site_type_symbol(element))
                c.add_data("atoms", _atom_site_description=CIF.atom_site_description(type))
                c.add_data("atoms", _atom_site_fragment=CIF.atom_site_fragment(order))
                c.add_data("atoms", _atom_site_fract_x=CIF.atom_site_fract_x(pos[0]))
                c.add_data("atoms", _atom_site_fract_y=CIF.atom_site_fract_y(pos[1]))
                c.add_data("atoms", _atom_site_fract_z=CIF.atom_site_fract_z(pos[2]))
        
        supercells = np.array(list(itertools.product((-1,0,1), repeat=3)))
        unit_repr = np.array([5,5,5], dtype=int)
        for (at1, at2), type in self.mof.bonds.items():
            try:
                atom1 = indices.index(at1)
                atom2 = indices.index(at2)
            except ValueError:
                return False
            label1 = labels[atom1]
            label2 = labels[atom2]
            test_coords = np.array([np.dot(i, self.cell) for i in fcoords[atom2] + supercells])
            coord = np.dot(fcoords[atom1], self.cell)
            dists = distance.cdist([coord], test_coords)
            dists = dists[0].tolist()
            dist = min(dists)
            image = dists.index(dist)
            sym = '.' if all([i==0 for i in supercells[image]]) else\
                    "1_%i%i%i"%(tuple(np.array(supercells[image], dtype=int) + unit_repr))
            c.add_data("bonds", 
                _geom_bond_atom_site_label_1=CIF.geom_bond_atom_site_label_1(label1))
            c.add_data("bonds", 
                _geom_bond_atom_site_label_2=CIF.geom_bond_atom_site_label_2(label2))
            c.add_data("bonds", 
                _geom_bond_distance=CIF.geom_bond_distance(dist))
            c.add_data("bonds", 
                _geom_bond_site_symmetry_2=CIF.geom_bond_site_symmetry_2(sym))
            c.add_data("bonds", 
                _ccdc_geom_bond_type=CIF.ccdc_geom_bond_type(CCDC_BOND_ORDERS[type]))
        
        file = open("%s.cif"%(c.name), "w")
        file.writelines(str(c))
        file.close()
        return True

    def pickle_prune(self):
        """Delete extra data from net - so that the pickle file does not 
        get too large."""
        del self.fragments
        del self.mof
        del self.main_sub_graph
        del self._coord_units
