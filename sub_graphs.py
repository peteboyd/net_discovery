import numpy as np
import itertools
import operator
from scipy.spatial import distance
from logging import debug
from faps import Structure
from function_switch import FunctionalGroupLibrary, FunctionalGroup
from SecondaryBuildingUnit import SBU
import pybel
import sys
pybel.ob.obErrorLog.StopLogging()

class SubGraph(object):

    def __init__(self, options=None, name="Default"):
        self.options = options
        self.name = name
        self._coordinates = None
        self._orig_index = []
        self._new_index = []
        self.bonds = {}

    def from_faps(self, struct, supercell=(1,1,1)):
        cell = struct.cell.cell
        inv_cell = np.linalg.inv(cell.T)
        size = len(struct.atoms)
        # number of cells
        try:
            sc = self.options.supercell
        except AttributeError:
            sc = supercell
        multiplier = reduce(operator.mul, sc, 1)
        self._coordinates = np.empty((size*multiplier, 3), dtype=np.float64)
        self.elements = range(size*multiplier)
        self._orig_index = range(size*multiplier)
        self._new_index = range(size*multiplier)
        supercell = list(itertools.product(*[itertools.product(range(j)) for j in
                    sc]))
        atom_size = 0
        for id, atom in enumerate(struct.atoms):

            for mult, scell in enumerate(supercell):
                atom_size += 1
                # keep symmetry translated index
                self._orig_index[id + mult * size] = id
                self.elements[id + mult * size] = atom.element    
                fpos = atom.ifpos(inv_cell) + np.array([float(i[0]) for i in scell])
                self._coordinates[id + mult * size][:] = np.dot(fpos, cell)
        self._new_index = range(atom_size)
        # shift the centre of atoms to the middle of the unit cell.
        supes = (cell.T * sc).T
        isupes = np.linalg.inv(supes.T)
        #self.debug("supercell")
        cou = np.sum(cell, axis=0)/2. 
        coa = np.sum(supes, axis=0)/2. 
        shift = coa - cou
        self._coordinates += shift
        # put all the atoms within fractional coordinates of the
        # supercell
        for i, cc in enumerate(self._coordinates):
            frac = np.array([k%1 for k in np.dot(isupes, cc)])
            self._coordinates[i] = np.dot(frac, supes)
        #self.debug("supercell")
        # create supercell bond matrix
        for bond, val in struct.bonds.items():
            for mult, scell in enumerate(supercell):
                b1 = bond[0] + mult*size 
                b2 = bond[1] + mult*size
                self.bonds[(b1,b2)] = val
        # shift the cell back??
        shift = cou - coa
        self._coordinates += shift
        #self.debug("supercell")

    def compute_bonds(self, cell):
        """Shifts bonds by a periodic boundary. This is so that the right
        atoms are bonded together in the supercell."""
        icell = np.linalg.inv(cell.T)
        for (b1, b2), val in self.bonds.items():
            # take each b1 atom and shift by the min img convention.
            # find min distances.
            f = np.dot(icell, self._coordinates[b1])
            # get all the images of the bonding atom
            b2_images = [i for i, j in enumerate(self._orig_index) 
                         if j == self._orig_index[b2]]
            assert b2 in b2_images
            # convert to fractional coordinates
            b2_fractals = np.array([np.dot(icell, i) for i in 
                                    self._coordinates[b2_images]])
            # shift fractionals to within a boundary box of b1
            shifted = b2_fractals + np.around(f - b2_fractals) 
            # gather distances
            dist = distance.cdist([f], shifted)[0]
            # find minimum distance index of all images
            id = np.where(dist==np.amin(dist))[0][0]
            bond = tuple([b1, b2_images[id]])
            if bond not in self.bonds.keys():
                self.bonds.pop((b1, b2))
                self.bonds[bond] = val

    def fragmentizer(self, at, r=[], q=[]):
        r.append(at)
        bonds = self.get_bonding_atoms([at])
        bonds = [i for i in bonds if i not in q + r]
        q += bonds
        if not q:
            return r
        else:
            return self.fragmentizer(q[0], r, q[1:])

    def set_elem_to_C(self):
        for id, atom in enumerate(self.elements):
            self.elements[id] = "C"

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

    def get_bonding_atoms(self, indices):
        ret_array = []
        for (i1, i2) in self.bonds.keys():
            if i1 in indices:
                if i2 not in indices:
                    ret_array.append(i2)
            if i2 in indices:
                if i1 not in indices:
                    ret_array.append(i1)
        return ret_array

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
            sub._dmatrix[y][x] = self.distances[array[y]][array[x]]
        sub.elements = [self[x] for x in array]
        sub._coordinates = np.array([self._coordinates[x] for x in array],
                                    dtype=np.float64)
        sub._orig_index = [self._orig_index[x] for x in array]
        sub._new_index = [self._new_index[x] for x in array]
        #sub.bonds = {(i1,i2):val for (i1, i2), val in self.bonds.items() if i1 in 
        #                sub._new_index and i2 in sub._new_index}
        sub.bonds = self.bonds.copy()
        return sub

    def __copy__(self):
        sub = SubGraph(self.options, name=self.name)
        try:
            sub._dmatrix = self._dmatrix.copy()
        except AttributeError:
            pass
        sub.elements = self.elements[:] 
        sub._coordinates = self._coordinates.copy() 
        sub._orig_index = self._orig_index[:] 
        sub._new_index = self._new_index[:]
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
        try:
            self._coordinates = np.vstack((self._coordinates, obj._coordinates))
        except ValueError:
            self._coordinates = obj._coordinates.copy()
        self._orig_index += obj._orig_index
        self._new_index += obj._new_index
        self.bonds.update(obj.bonds)
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

    def __init__(self, options, name="Default"):
        self.options = options
        self.name = name
        self._elements = []
        self._coordinates = np.array([]) 
        self._orig_index = []
        self._new_index = []
        self.bonds = {}

    def to_mol(self):
        """Convert fragment to a string in .mol format"""
        header = "Organic\n %s\n\n"%(self.name)
        counts = "%3i%3i%3i%3i%3i%3i%3i%3i%3i 0999 V2000\n"
        atom_block = "%10.4f%10.4f%10.4f %3s%2i%3i%3i%3i%3i%3i%3i%3i%3i%3i%3i%3i\n"
        bond_block = "%3i%3i%3i%3i%3i%3i%3i\n"
        mol_string = header
        mol_string += counts%(len(self), len(self.bonds.keys()), 0, 0, 0, 0, 0, 0, 0)
        atom_order = []
        for i in range(len(self)):
            pos = self._coordinates[i]
            mol_string += atom_block%(pos[0], pos[1], pos[2], self.elements[i],
                                      0,0,0,0,0,0,0,0,0,0,0,0)
            atom_order.append(self._orig_index[i])

        for bond, type in self.bonds.items():
            ind1 = atom_order.index(bond[0]) + 1
            ind2 = atom_order.index(bond[1]) + 1
            b_order = 4 if type == 1.5 else type
            mol_string += bond_block%(ind1, ind2, b_order, 0, 0, 0, 0)
        return mol_string
    
    def update_bonds(self, bond_dic):
        for (i, j), type in bond_dic.items():
            if i in self._orig_index and j in self._orig_index:
                self.bonds.update({(i,j):type})

    def inchikey(self):
        """Return the inchikey"""
        string = self.to_mol()
        conv = pybel.ob.OBConversion()
        conv.SetOutFormat('inchikey')
        mol = pybel.readstring('mol', string)
        return conv.WriteString(mol.OBMol).strip()
