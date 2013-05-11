#!/usr/bin/env python

"""
GenStruct -- Generation of Structures for fapping.
"""

__version__ = "$Revision$"

import subprocess
import textwrap
import sys
import math
import re
import os
import io
import itertools
import ConfigParser
import functional
from config import Options
from operations import *
import numpy as np
from numpy import array, pi
from elements import WEIGHT
from bookkeeping import * 
from random import random, uniform, randrange, choice
from datetime import date
from logging import warning, debug, error, info, critical
from scipy.spatial import distance
from itertools import combinations

class Bond(object):
    """ Bonds between atoms """
    def __init__(self, bond_from, bond_to, type, distance=None):
        self.frm = bond_from
        self.to = bond_to # can be Atom or ConnectPoint
        self.type = type
        self.vector = bond_to.coordinates[:3] - bond_from.coordinates[:3]
        # note the distance will change for the ConnectPoint bond
        # once the Atom is bonded to another building unit.
        if distance is not None:
            self.distance = distance
        else:
            self.distance = np.linalg.norm(self.vector)

class Atom_gen(object):
    """
    Contains atomistic information read in from an input file
    """
    def __init__(self, text=None):

        self.coordinates = np.ones(4)

        if text is None:
            self.element = "X"
            self.mass = 0.
            self.force_field_type = None

        else:
            text = text.split()
            # element type (e.g. C)
            self.element = text[0].strip()
            # mass of element
            self.mass = WEIGHT[self.element]
            # cartesian coordinates (x y z)
            self.coordinates[:3] = array([float(i) for i in text[2:5]],
                                 dtype=float)
            # force field atom type read in from an input file
            self.force_field_type = text[1]
        # fractional coordinates (a b c)
        self.scaled_coords = None
        # the bu index this atom belongs to
        self.bu_index = None
        # flag for if the building unit is a metal
        self.bu_metal = False
        # index of functional group (also functions as a flag)
        self.fnl_group_index = None
        # list of bonds which connect atoms 
        self.bonds = []
        # This will reference a connect point of a building unit
        # if it is a connecting atom
        self.connectivity = None
        # index as listed in the Structure
        self.index = 0
        # index as listed in the Building Unit
        self.internal_index = 0

    def __str__(self):
        
        return "Atom %3i, %2s, location: (%9.5f, %9.5f, %9.5f)\n"%(
                tuple([self.index, self.element] + 
                    [i for i in self.coordinates[:3]]))\
                +"Belongs to: %s\nmetal: %s\n"%(str(self.bu_index),
                        str(self.bu_metal))\
                +"Connect atom: %s\n"%str(self.connectivity)


class BuildingUnit(object):
    """
    Contains collections of Atom objects and connectivity
    information for building MOFs
    """

    def __init__(self, name=None, items=[]):
        items = dict(items)
        self.name = name 
        # this is a global index which is reported in the
        # structure name.
        self.index = int(items.pop('index'))
        # keep track of each building unit internally with it's own
        # index.
        self.internal_index = 0
        # keep track of what order the building unit was placed in the
        # MOF
        self.order = 0
        # topology
        self.topology = items.pop('topology')
        # track if metal building unit
        self.metal = items.pop('metal').lower() == "true"

        # in case of special conditions
        if items.has_key('parent'):
            self.parent = items.pop('parent')
        else:
            self.parent = None
        
        self.functional_groups = [] # list of functional groups.
        # Topology info:
        # points of connection for other building units
        self.build_connect_points(items.pop('connectivity'))

        # set up atoms and their connectivity
        self.build_atoms(items.pop('coordinates'), 
                          items.pop('table'))

        # determine bonding constraints
        self.build_bond_const(items.pop('bond_constraints'))

        # Special Building Units for multiple versions of the
        # same building unit
        self.specialbu = []

        # centre of mass
        self.calculate_COM()
        
        # check for angles, this only applies to linear molecules which can
        # be rotated
        if items.has_key('angles'):
            self.angles = items.pop('angles')
        else:
            self.angles = None
            
    def build_atoms(self, coordinates, table):
        """
        Read in coordinates and table strings to build the 
        atoms and connectivity info.
        """
        self.atoms, self.bonds = [], []
        ind = 0
        for atom in coordinates.splitlines():
            atom = atom.strip()
            if not atom:
                continue
            self.atoms.append(Atom_gen(atom))
            self.atoms[-1].index = ind
            self.atoms[-1].internal_index = ind
            self.atoms[-1].bu_index = self.index
            self.atoms[-1].bu_metal = self.metal
            ind += 1

        table = table.strip()
        # self.connect_points must be populated before doing this.
        for bond in table.splitlines():
            bond = bond.strip().split()
            # add the bonding information
            # first two cases are for bonding to connecting points
            if "c" in bond[0].lower():
                connect_ind = int(bond[0].lower().strip('c'))
                atom_ind = int(bond[1])
                self.bonds.append(Bond(
                    self.connect_points[connect_ind],
                    self.atoms[atom_ind], bond[2]))
                # add the Atom to the connect_point
                self.connect_points[connect_ind].atoms.append(
                        self.atoms[atom_ind])
                self.atoms[atom_ind].connectivity = connect_ind

            elif "c" in bond[1].lower():
                # subtract 1 since the input file starts at 1
                connect_ind = int(bond[1].lower().strip('c'))-1
                atom_ind = int(bond[0])
                self.bonds.append(Bond(
                    self.atoms[atom_ind],
                    self.connect_points[connect_ind],
                    bond[2]))
                # add the Atom to the connect_point
                self.connect_points[connect_ind].atoms.append(
                        self.atoms[atom_ind])
                self.atoms[atom_ind].connectivity = connect_ind

            else:
                # add bonding to atoms
                # add the index of the atom it is bonded with
                self.atoms[int(bond[0])].bonds.append(
                        int(bond[1]))
                self.atoms[int(bond[1])].bonds.append(
                        int(bond[0]))
                self.bonds.append(Bond(
                     self.atoms[int(bond[0])], 
                     self.atoms[int(bond[1])], bond[2]))
        
    def build_connect_points(self, connectivity):
        """
        Read connectivity string to build the connectivity
        info.
        """
        connectivity = connectivity.strip()
        self.connect_points = []
        for connpt in connectivity.splitlines():
            order = len(self.connect_points)
            self.connect_points.append(ConnectPoint(connpt))
            self.connect_points[-1].order = order
            if self.metal:
                self.connect_points[-1].metal = True

    def build_bond_const(self, constraints):
        """
        Read in bond constraint info in the order:
        [connect_point.index, connect_point.special]
        """
        constraints = constraints.strip()
        # this assumes connect_points is already established
        for pt in self.connect_points:
            pt.constraint = None
        for const in constraints.splitlines():
            const = const.strip().split()
            for pt in self.connect_points:
                if pt.index == int(const[0]):
                    pt.constraint = int(const[1])

    def snap_to(self, self_point, connect_point):
        """
        adjusts atomic coordinates to orient correctly with
        connect_point
        """
        angle = calc_angle(self_point.para[:3], -connect_point.para[:3])
        # in case the angle is zero
        if np.allclose(angle, 0.):
            trans_v = connect_point.coordinates - \
                      self_point.coordinates

            for atom in self.atoms:
                atom.coordinates += trans_v
            for cp in self.connect_points:
                cp.coordinates += trans_v

            return

        axis = np.cross(self_point.para[:3], connect_point.para[:3])
        # check for parallel parameters
        if np.allclose(axis, np.zeros(3), atol=0.02):
            # They are aligned parallel, must be rotated 180 degrees.
            axis = self_point.perp[:3]
            angle = np.pi
        
        axis = axis / length(axis)

        if np.allclose(self.COM, np.zeros(3)):
            pt = None
        else:
            pt = self.COM[:]
        R = rotation_matrix(axis, angle, point=pt)
        # TODO(pboyd): applying this transform one by one is 
        # really inefficient, and should be done to a giant array
        trans_v = connect_point.coordinates - np.dot(R, 
                               self_point.coordinates)
        
        for atom in self.atoms:
            atom.coordinates = np.dot(R, atom.coordinates) + trans_v

        for cp in self.connect_points:
            cp.coordinates = np.dot(R, cp.coordinates) + trans_v
            # apply rotation matrix to vectors
            cp.para[:3] = np.dot(R[:3,:3], cp.para[:3])
            cp.perp[:3] = np.dot(R[:3,:3], cp.perp[:3])
            
    def align_to(self, self_point, connect_point):
        """
        aligns two building units along their perp vectors
        """
        angle = calc_angle(self_point.perp[:3], connect_point.perp[:3])
        if np.allclose(angle, 0.):
            return
        axis = connect_point.para[:3] 
        R = rotation_matrix(axis, angle, point=
                            self_point.coordinates)
        # NOTE: the default min tolerance was set to 0.06 because when
        # building CuBTC lower values would result in the opposite
        # rotation.  ie. the initial rotation carried out with R resulted
        # in the two perp vectors to be aligned with a error of 0.06 radians
        # although with the following corrective measures, it typically
        # goes down to 6e-4 rad
        tol = min(angle, 0.06)
        check_vect = np.dot(R[:3,:3], self_point.perp[:3])
        if not np.allclose(calc_angle(check_vect, 
                           connect_point.perp),0.,
                           atol=tol):
            init_angle = calc_angle(check_vect, connect_point.perp)
            debug("initial angle: %f"%init_angle)
            angle = -angle 
            R = rotation_matrix(axis, angle, 
                                point=self_point.coordinates)
            
            check_vect = np.dot(R[:3,:3], self_point.perp[:3])
            if calc_angle(check_vect, connect_point.perp) > init_angle:
                # do some extra rotation.
                R = np.dot(rotation_matrix(axis, init_angle,
                                    point=self_point.coordinates),
                           rotation_matrix(axis, -angle,
                                    point=self_point.coordinates))
                check_vect = np.dot(R[:3,:3], self_point.perp[:3])
                
                
        debug("final angle: %f"%calc_angle(check_vect, connect_point.perp))

        for atom in self.atoms:
            atom.coordinates = np.dot(R, atom.coordinates)
            #atom.coordinates = np.dot(atom.coordinates, R)
        for cp in self.connect_points:
            cp.coordinates = np.dot(R, cp.coordinates)
            cp.para[:3] = np.dot(R[:3,:3], cp.para[:3])
            cp.perp[:3] = np.dot(R[:3,:3], cp.perp[:3])

    def calculate_COM(self):
        self.COM = \
            np.average(array([atom.coordinates[:3] for atom in self.atoms]), 
                       axis=0, 
                       weights=array([atom.mass for atom in self.atoms]))
        return
        
    def __str__(self):
        line = "Building Unit from Genstruct: %s\n"%self.name
        for atom in self.atoms:
            line += "%3s%9.3f%7.3f%7.3f"%tuple([atom.element] + 
                            list(atom.coordinates[:3]))
            line += " UFF type: %s\n"%(atom.force_field_type)
        line += "Connectivity Info:\n"
        for con in self.connect_points:
            line += "%3i%9.3f%7.3f%7.3f"%tuple([con.index] +
                        list(con.coordinates[:3]))
            line += " Special bonding: %5s"%str(con.special)
            line += " Symmetry flag: %s\n"%str(con.symmetry)
        return line

    def find_bonds(self, atom):
        """Returns a list of Bonds if the atom belongs to them."""
        blist = []
        for bond in self.bonds:
            if bond.frm == atom or bond.to == atom:
                blist.append(bond)
        return blist
    
    def functionalize(self, functional_group, hydrogen):
        """Append the functional group to the building unit atoms."""
        append_point = hydrogen
        self.atoms = (self.atoms[:hydrogen-1] + functional_group.atoms
                      + self.atoms[hydrogen:])
        # remove bond to hydrogen
        remove = []
        for idx, bond in enumerate(self.bonds):
            if bond.to.index == hydrogen or bond.frm.index == hydrogen:
                remove.append(idx)
        for i in remove:
            self.bonds.pop(i)
        for atom in functional_group.atoms:
            if atom.connectivity:
                fnl_bonding_atom = atom
                
        # add bond to self.bonds
        self.bonds.append(Bond(functional_group.MOF_atom,
                               fnl_bonding_atom,
                               "S",
                               distance=functional_group.bond_length))

class ConnectPoint(object):
    """
    Contains one point of connection, including associated
    parallel and perpendicular alignment vectors

    In the input file the connect_point should read:
    [index] point[x,y,z] vector[x,y,z] vector[x,y,z] [integer] [integer]
    """

    def __init__(self, text=None):
        if text is None:
            self.index = 0  # index for determining bonding data
            self.coordinates = np.ones(4)   # the location of the connect point
            self.para = np.ones(4) # parallel bonding vector
            self.perp = np.ones(4) # perpendicular bonding vector
            self.atoms = []     # store the atom(s) which connect to this point
            self.special = None # for flagging any special type of bond
        else:
            text = text.strip().split()
            # index will be important for flagging specific bonding
            # and obtaining the connectivity table at the end.
            self.index = int(text[0])
            # point of connection
            self.coordinates = np.ones(4)
            self.coordinates[:3] = array([float(x) for x in text[1:4]])
            # parallel alignment vector
            self.para = np.ones(4)
            self.para[:3] = array([float(z) for z in text[4:7]])
            self.para[:3] = self.para[:3] / length(self.para[:3])
            # perpendicular alignment vector
            self.perp = np.ones(4)
            self.perp[:3] = array([float(y) for y in text[7:10]]) 
            self.perp[:3] = self.perp[:3] / length(self.perp[:3])

        self.order = 0  # keep track of what list order the CP is in the BU
        self.metal = False  # for flagging if the associated BU is metal
        self.bu_order = 0   # for keeping track of what order the BU is 
        # list of bonding atoms.  In most cases this will
        # be only one, but there are exceptions (Ba MOF)
        self.atoms = []
        # flag for if the point has been bonded
        self.bonded = False
        # flag for if the bond is found after the bu is added
        self.bond_from_search = False
        try:
            self.special = int(text[11])
        except:
            self.special = None

        # constraint constrains this connectivity point to a 
        # particular special bond index
        self.constraint = None

        # symmetry type flag
        # other connect_points which it is equivalent (of the same 
        # building unit) will have the same self.equiv value
        try:
            self.symmetry = int(text[10])
        except:
            self.symmetry = 1
        # store what symmetry type of bond has been formed, this should
        # include both the internal indices of the bu and the symmetry labels
        self.bond_label = None 

    def __str__(self):
        if not self.special:
            spec = 0
        else:
            spec = self.special
        if not self.constraint:
            const = 0
        else:
            const = self.constraint
        string = "Bond %i (%5.2f, %5.2f, %5.2f)\n"%tuple([self.index] +
                list(self.coordinates[:3]))
        string += "Symmetry: %2i, Special: %2i, Constraint: %2i"%(
                self.symmetry, spec, const)
        return string

class Structure(object):
    """
    Contains building units to generate a MOF
    """
    def __init__(self):
        self.building_units = []
        self.cell = Cell()
        self.xyz_lines = ""  # for debugging to a .xyz file
        self.natoms = 0
        # Store global bonds here
        self.bonds = []
        self.sym_id = []
        self.debug = True   # have this parsed from command line
        self.directives = [] # keep track of bonding directives
        self.connecting_bonds = []  # list of Bonds containing connecting atoms.

    def debug_xyz(self):
        if self.debug:
            cellformat = "H%12.5f%12.5f%12.5f " + \
                    "atom_vector%12.5f%12.5f%12.5f\n"
            bondformat = "H%12.5f%12.5f%12.5f " + \
                    "atom_vector%12.5f%12.5f%12.5f " + \
                    "atom_vector%12.5f%12.5f%12.5f\n"
            atomformat = "%s%12.5f%12.5f%12.5f\n"

            lines = []
            [lines.append(cellformat%tuple(list(self.cell.origin[ind]) + 
                          list(self.cell.lattice[ind]))) for ind in range(
                          self.cell.index)]
            for bu in self.building_units:
                [lines.append(bondformat%tuple(list(c.coordinates[:3]) +
                    list(c.para[:3]) + list(c.perp[:3]))) for c in bu.connect_points]
                [lines.append(atomformat%tuple([a.element] + 
                    list(a.coordinates[:3]))) for a in bu.atoms]

            n = len(lines)

            self.xyz_lines += "%5i\ndebug\n"%n
            self.xyz_lines += "".join(lines)

    def insert(self, bu, bond, add_bu, add_bond):

        self.building_units.append(add_bu)   # append new BU to existing list
        self.debug_xyz()  # store coordinates
        add_bu.snap_to(add_bond, bond)  # align the para vectors
        order = len(self.building_units) - 1  # order of the new building unit
        add_bu.order = order
        # asign the bu order to the connect points in the bu
        for cp in add_bu.connect_points:
            cp.bu_order = order
        self.debug_xyz()  # store coordinates
        add_bu.align_to(add_bond, bond)  # align the perp vectors
        # re-order atom indices within bu.atoms
        # to coincide with connectivity
        
        for atom in add_bu.atoms:
            # adjust the atom index
            atom.index += self.natoms
            # adjust the bonding indices for each atom
            for idx, bnd in enumerate(atom.bonds):
                atom.bonds[idx] = bnd + self.natoms
        # update the number of atoms in the structure
        self.natoms += len(add_bu.atoms)
        # introduce bond between joined connectivity points
        self.update_connectivities(bu, bond, add_bu, add_bond)
        self.store_building_unit_bonds(add_bu)
        # check for bonding between other existing building units
        self.bonding()
        self.debug_xyz()  # store coordinates
        self.directives.append((bu.order, bond.index,
                                add_bu.index, add_bond.index))
        return

    def bonding(self):
        """
        Check for new periodic boundaries, bonds using existing periodic boundaries,
        or local bonds.
        """
        connect_pts = [cp for bu in self.building_units for 
                       cp in bu.connect_points]
        bond_combos = itertools.combinations(connect_pts, 2)
        for cp1, cp2 in bond_combos:
            ibu1 = cp1.bu_order 
            bu1 = self.building_units[ibu1]
            ibu2 = cp2.bu_order
            bu2 = self.building_units[ibu2]
            if valid_bond(cp1, cp2):
                if self.is_aligned(cp1, cp2):
                    # determine the vector between connectpoints
                    dvect = (cp2.coordinates[:3] - 
                             cp1.coordinates[:3])
                    # get a shifted vector to see if the bond will
                    # be local
                    svect = self.cell.periodic_shift(dvect.copy())       
                    # test for local bond
                    debug("comparing (%9.5f, %9.5f, %9.5f) and "
                          %(tuple(cp1.para[:3]))+"(%9.5f, %9.5f, %9.5f)"
                          %(tuple(cp2.para[:3])))
                    if np.allclose(length(svect), 0., atol=0.4):
                        debug("local bond found between "
                              +"#%i %s, bond %i and "
                              %(ibu1, bu1.name, cp1.index)
                              +"#%i %s, bond %i."
                              %(ibu2, bu2.name, cp2.index))
                        # join two bonds
                        self.update_connectivities(bu1, cp1, bu2, cp2)
                        cp1.bond_from_search = True
                        cp2.bond_from_search = True
                    # check for valid periodic vector
                    elif self.cell.valid_vector(dvect):
                        debug("periodic vector found: " 
                        +"(%9.5f, %9.5f, %9.5f)"%(
                            tuple(dvect.tolist()))
                        +" between #%i %s, bond %i"%(
                            ibu1, bu1.name, cp1.index)
                        +" and #%i %s, bond %i"%(
                            ibu2, bu2.name, cp2.index))
                        self.cell.origin[self.cell.index] =\
                                cp1.coordinates[:3].copy()
                        self.cell.add_vector(dvect)
                        self.update_connectivities(bu1, cp1, bu2, cp2)
                        cp1.bond_from_search = True
                        cp2.bond_from_search = True           
                    else:
                        debug("No bonding found with "
                        +"original (%9.5f, %9.5f, %9.5f),"
                        %(tuple(dvect.tolist()))+
                        " shifted (%9.5f, %9.5f, %9.5f)"
                        %(tuple(svect.tolist())))

    def is_aligned(self, cp1, cp2):
        """Return True if the connect_points between bu1 and bu2 are
        parallel.

        """
        # FIXME(pboyd): note a tolerance of 0.2 corresponds to 30 degrees!
        # this has been lowered to 0.05 and should be tested across a
        # range of existing mofs.
        if parallel(-cp1.para, cp2.para, tol=0.05):
            if parallel(cp1.perp, cp2.perp, tol=0.05):
                return True
            elif antiparallel(cp1.perp, cp2.perp, tol=0.05):
                return True
        return False

    def overlap_bu(self, bu):
        """
        Check the bu supplied for overlap with all other atoms
        """
        # scale the van der waals radii by sf
        sf = 0.4
        # NEED TO DEBUG here!
        atomlist = [atom for tbu in self.building_units for atom in tbu.atoms]
        for atom in bu.atoms:
            elem, coords = self.min_img_shift(atom=atom.coordinates)
            # distance checks
            #coordlist = [list(atom.coordinates[:3])]
            #for coord in coords.tolist():
            #    coordlist.append(coord)
            #atmlist = [atom.element] + elem
            #xyz_file(atoms=atmlist, coordinates=coordlist)
            distmat = distance.cdist([atom.coordinates[:3]], coords)
            # check for atom == atom, bonded
            excl = [atom.index] + [idx for idx in atom.bonds]

            for idx, dist in enumerate(distmat[0]):
                if idx not in excl:
                    if dist < (Radii[atom.element]+ Radii[elem[idx]])*sf:
                        return True
        return False

    def min_img_shift(self, atom=np.zeros(3)):
        """
        shift all atoms in the building units to within the periodic
        bounds of the atom supplied
        """
        mof_coords = np.zeros((self.natoms, 3))
        elements = []
        if self.cell.index:
            fatom = np.dot(atom[:3], self.cell.ilattice)
            atcount, max = 0, 0
            for bu in self.building_units:
                elements += [axx.element for axx in bu.atoms]
                max = atcount + len(bu.atoms)
                fcoord = array([np.dot(a.coordinates[:3], self.cell.ilattice)
                            for a in bu.atoms])
                # shift to within the pbc's of atom coords
                rdist = np.around(fatom-fcoord)
                if self.cell.index < 3:
                    # orthogonal projection within cell boundaries
                    shift = np.dot(rdist, 
                        self.cell.lattice[:self.cell.index])
                    at = (array([a.coordinates[:3] for a in bu.atoms])
                      + shift)
                else:
                    at = np.dot(fcoord+rdist, self.cell.lattice[:self.cell.index])
                mof_coords[atcount:max] = at
                atcount += len(bu.atoms)
            # INCLUDE ROUTINE TO APPEND SHIFTED FUNCTIONAL GROUPS TO mof_coords
            for bu in self.building_units:
                if bu.functional_groups:
                    pass
                
        else:
            elements += [axx.element for bux in self.building_units for
                         axx in bux.atoms]
            mof_coords = array([a.coordinates[:3] for s in 
                                self.building_units for a in s.atoms])

        return elements, mof_coords 

    def atom_min_img_shift(self, pos1, pos2):
        """
        shift x,y,z position from pos2 to within periodic boundaries
        of pos1.  Return the shifted coordinates.
        """
        if self.cell.index:
            fpos1 = np.dot(pos1[:3], self.cell.ilattice)
            fpos2 = np.dot(pos2[:3], self.cell.ilattice)
            # shift to within the pbc's of atom coords
            rdist = np.around(fpos1-fpos2)
            if self.cell.index < 3:
                # orthogonal projection within cell boundaries
                shift = np.dot(rdist, 
                        self.cell.lattice[:self.cell.index])
                return pos2[:3] + shift
            else:
                return np.dot(fpos2+rdist, self.cell.lattice[:self.cell.index])

        else:
            return pos2[:3]

    def saturated(self):
        """ return True if all bonds in the structure are bonded """
        unsat_bonds = len([1 for bu in self.building_units for cp in
                           bu.connect_points if not cp.bonded])
        if unsat_bonds == 0:
            return True
        return False

    def update_connectivities(self, bu, cp, add_bu, add_cp):
        """ 
        update the atom.bond and bu.bonds when a bond is formed between
        two building units.
        """
        # bond tolerance between two atoms.  This will only be relevant
        # when there is more than one atom associated with a particular
        # bond. Eg. Ba2+
        sf = 0.6
        # determine the symmetry label
        symmetry_label = [(bu.internal_index, cp.symmetry), 
                          (add_bu.internal_index, add_cp.symmetry)]
        symmetry_label.sort()
        symmetry_label = tuple(symmetry_label)
        cp.bonded, add_cp.bonded = True, True
        cp.bond_label, add_cp.bond_label = symmetry_label, symmetry_label
        for atm in cp.atoms:
            for atm2 in add_cp.atoms:
                # periodic shift atm2 in terms of atm to calculate distances
                shiftcoord = self.atom_min_img_shift(atm.coordinates, 
                                                    atm2.coordinates)
                dist = length(atm.coordinates, shiftcoord)
                if dist < (Radii[atm.element] + Radii[atm2.element])*sf:
                    # append self.bonds with both atoms.
                    atm.bonds.append(atm2.index)
                    atm2.bonds.append(atm.index)
                    # append to bonds
                    # bond type "single" for now...
                    self.bonds.append((atm.index, atm2.index, "S", dist))
                    # special connecting bonds require reference to
                    # atom instances.
                    self.connecting_bonds = Bond(atm, atm2, "S", dist)
                    #TODO(pboyd): add check for if no Bond is associated
                    # with the connect_point, in which case raise a 
                    # warning. (something wrong with the input)

    def store_building_unit_bonds(self, add_bu):
        """Store the bonds local to the building units to the structures
        bonding memory.  A bit redundant, but makes it easier when files
        are written.

        """
        for bond in add_bu.bonds:
            # check if bonds are to Atom()s or to ConnectPoint()s
            if isinstance(bond.frm, ConnectPoint) or \
                    isinstance(bond.to, ConnectPoint):
                pass
            else:
                self.bonds.append((bond.frm.index, bond.to.index,
                                   bond.type, bond.distance))

    def get_scaled(self):
        """
        Add scaled coordinates to all of the atoms in the structure.
        This should be done once all of the periodic boundaries are set.
        """
        for bu in self.building_units:
            for atm in bu.atoms:
                atm.scaled_coords = self.cell.get_scaled(atm.coordinates)
                # note I do not shift to within periodic boundaries yet.
                # this should be done later.
        return

    def finalize(self, base_building_units, outdir="", csvfile=None):
        """Write final files."""
        # determine the bonding sequence.
        self.cell.reorient()
        # check if lattice needs to be inverted
        if self.cell.lattice[2][2] < 0.:
            # invert the cell
            self.cell.lattice[2][2] = -1. * self.cell.lattice[2][2]
            # do a 1-scaled_coords.
            for bu in self.building_units:
                for atom in bu.atoms:
                    atom.scaled_coords = 1 - atom.scaled_coords
        self.cell.get_inverse()
        self.get_cartesians()
        self.recalc_bond_vectors()
        met_lines = ""
        org_lines = ""
        metal_ind, organic_ind = [], []
        for bu in base_building_units:
            if bu.metal:
                metal_ind.append(bu.index)
                topology = bu.topology
            else:
                organic_ind.append(bu.index)
        for met in set(metal_ind):
            met_lines += "_m%i"%(met)
        for org in set(organic_ind):
            org_lines += "_o%i"%(org)
        # temp fix for now
        if len(set(organic_ind)) == 1:
            org_lines += "_o%i"%(org)
            
        basename = "str" + met_lines + org_lines + "_%s"%(topology)
        filename = outdir + basename 

        cif_file = CIF(self, sym=True)
        cif_file.write_cif(filename)

        # hack job on the csvfile... didn't plan ahead on this one.
        if csvfile:
            hm_name = cif_file.symmetry.get_space_group_name()
            sym_number = cif_file.symmetry.get_space_group_number()
            csvfile.add_data(
                    MOFname = basename,
                    metal_index = metal_ind[0],
                    organic_index1 = organic_ind[0],
                    organic_index2 = organic_ind[1],
                    h_m_symmetry_name = hm_name,
                    symmetry_number = sym_number,
                    build_directive = self.directives)     

    def recalc_bond_vectors(self):
        """Re-calculates all bond vectors after all rotations are completed."""
        for building_unit in self.building_units:
            for bond in building_unit.bonds:
                # don't care about atom --> connect point,
                # only atom --> atom bonds.
                if isinstance(bond.frm, Atom_gen) and isinstance(bond.to, Atom_gen):
                    bond.vector = bond.frm.coordinates[:3] - bond.to.coordinates[:3]
                    
    def get_cartesians(self):
        """Convert fractional coordinates to cartesians.  This is necessary
        when the cell vectors are adjusted.
        
        """
        for building_unit in self.building_units:
            for atom in building_unit.atoms:
                atom.coordinates = np.dot(atom.scaled_coords, self.cell.lattice)
                

class Cell(object):
    """
    Contains vectors which generate the boundary conditions
    for a particular structure
    """
    def __init__(self):
        self.lattice = np.identity(3) # lattice vectors
        self.params = np.zeros(6) # a, b, c, alpha, beta, gamma
        self.ilattice = np.identity(3) # inverse of lattice
        self.nlattice = np.identity(3) # normalized vectors
        self.olattice = np.identity(3) # normalized orthogonal vectors
        self.index = 0  # keep track of how many vectors have been added
        self.origin = np.zeros((3,3)) # origin for cell vectors

    def get_inverse(self):
        """ Calculates the inverse matrix of the lattice"""
        M = np.matrix(self.lattice)
        self.ilattice = array(M[:self.index].I)
    
    def valid_vector(self, vector):
        """ 
        Checks to see if a vector can be added to the periodic
        boundary conditions.
        """
        normvect = unit_vector(vector[:3])

        for cellv in self.nlattice[:self.index]:
            if np.allclose(np.dot(cellv, normvect), 1, atol=0.1):
                debug("vector a linear combination of existing"
                           +" boundaries")
                return False
            #check for co-planar vector
        if self.index == 2:
            if self.planar(normvect):
                debug("vector a linear combination of existing"
                           +" boundaries")
                return False

        elif self.index == 3:
            return False
        
        return True

    def planar(self, vector):
        test = np.dot(vector, np.cross(self.nlattice[0], 
                                       self.nlattice[1]))
        return np.allclose(test, 0)

    def add_vector(self, vector):
        """Adds a vector to the cell"""
        self.lattice[self.index,:] = vector.copy()
        self.nlattice[self.index,:] = unit_vector(vector)
        # get orthogonal projection of vector
        self.store_orthogonal(vector)
        self.index += 1
        self.get_inverse()
        
    def store_orthogonal(self, v):
        """Store the orthogonal projection of the vector in an array."""
        vector = v.copy()
        # project
        for orthog_vect in self.olattice[:self.index]:
            vector = vector - project(vector, orthog_vect)
        vector = unit_vector(vector)
        self.olattice[self.index,:] = vector

    def periodic_shift(self, vector):
        """
        Shifts a vector to within the bounds of the periodic vectors
        """
        if self.index:
            # get fractional of vector
            proj_vect = np.dot(vector[:3],
                                self.ilattice)
            proj_vect = np.rint(proj_vect)

            shift_vect = np.dot(proj_vect, self.lattice[:self.index])

            # convert back to cartesians
            return (vector - shift_vect)

        return vector

    def get_scaled(self, vector):
        """
        Only applies if the periodic box is fully formed.
        """
        if self.index == 3:
            return np.dot(vector[:3], self.ilattice)

        return np.zeros(3)
    
    def reorient(self):
        """Re-orients the cell along the convention:
        a --> x axis
        b --> xy plane
        c --> the rest
        
        """
        xaxis = np.array([1.,0.,0.])
        yaxis = np.array([0.,1.,0.])
        # first: rotation to the x-axis
        x_rotangle = calc_angle(self.lattice[0], xaxis)
        x_rotaxis = np.cross(self.lattice[0], xaxis)
        x_rotaxis = (x_rotaxis)/length(x_rotaxis)
        RX = rotation_matrix(x_rotaxis, x_rotangle)
        # note the order of the dot product is reversed than compared with the
        # rotation of the building units.
        self.lattice = np.dot(self.lattice, RX[:3,:3])
        
        # second: rotation to the xy - plane
        projx_b = self.lattice[1] - project(self.lattice[1], xaxis)
        xy_rotangle = calc_angle(projx_b, yaxis)
        xy_rotaxis = xaxis
        RXY = rotation_matrix(xy_rotaxis, xy_rotangle)
        # test to see if the rotation is in the right direction
        testvect = np.dot(projx_b, RXY[:3,:3])
        testangle = calc_angle(yaxis, testvect)
        if not np.allclose(testangle, 0., atol = 1e-3):
            RXY = rotation_matrix(-xy_rotaxis, xy_rotangle)
        self.lattice = np.dot(self.lattice, RXY[:3,:3])
        return
    

class Database(list):
    """
    Reads in a set of Building Units from an input file
    """

    def __init__(self, filename):
        self.readfile(filename)
        # get extension of file name without leading directories.
        self.extension = filename.split('/')[-1]
        self.extension = self.extension.split(".dat")[0]
            
    def readfile(self, filename):
        """
        Populate the list with building units from the
        input file.
        """
        # Multidict is in bookkeeping.py to account for duplicate
        # names but mangles all the names up, so i will implement if
        # needed, but otherwise we'll just use neat 'ol dict
        #file = ConfigParser.SafeConfigParser(None, Multidict)
        file = ConfigParser.SafeConfigParser()
        file.read(filename)
        for idx, building_unit in enumerate(file.sections()):
            self.append(BuildingUnit(
                                    name=building_unit,
                                    items=file.items(building_unit)))
            self[-1].internal_index = idx
        # special considerations for linked building units
        for ind, bu in enumerate(self):
            if bu.parent:
                # link up this building unit with it's parent
                parent = [i for i in self 
                          if i.name == bu.parent]
                if len(parent) != 1:
                    # raise error
                    error("Multiple building units with the same name!")
                parent[0].specialbu.append(deepcopy(bu))
                self.pop(ind)
      
class Generate(object):
    """
    Algorithm for generating MOFs

    """

    def __init__(self, building_unit_database, num=3):
        self.bu_database = building_unit_database
        # select 1 metal and 2 organic linkers to mix
        # for the sampling.
        self.outdir = "output." + building_unit_database.extension + "/"
        # store the database of functional groups in case of functionalization.
        self.functional_groups = FunctionalGroupDatabase('functional_groups.lib')
        # include check to see if CSV is requested
        self.csv = CSV()
        self.csv.set_name(building_unit_database.extension)
        # place under write cif routine
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        combinations = self.set_combinations(num)
        # filter out multiple metal combo's
        combinations = self.filter(combinations)
        time = Time()
        time.timestamp()
        info("Genstruct will try %i combinations"%(len(combinations)))
        for combo in combinations:
            building_units = [building_unit_database[i] for i in combo]
            info("Trying combo, %s, %s, %s"%tuple([i.name for i in building_units]))
            self.exhaustive_sampling(building_units)
        self.csv.write_file()
        time.timestamp()
        info("Genstruct finished! timing reports %f seconds."%(time.timer))

    def set_combinations(self, num):
        """ Generates all combinations of a database with length n"""
        indices = [i for i in range(len(self.bu_database))]
        return list(itertools.combinations_with_replacement(indices, num))
    
    def filter(self, list):
        return [j for j in list if self.metcount(j) == 1]

    def metcount(self, combo):
        return len([i for i in combo if self.bu_database[i].metal])

    def generate_from_directives(self, directives):
        """Builds a MOF based on directives as tuples, starting with the
        initial insert.
        """
        return

    def expand_bu_list(self, bu_db):
        """Return a list of building units expanded out and deep-copied."""
        bu_list = []
        for bu in bu_db:
            bu_list.append(deepcopy(bu))
            # check for special building units nested in the bu.
            if bu.specialbu:
                for i in bu.specialbu:
                    bu_list.append(deepcopy(i))
        return bu_list
    
    def assign_internal_indices(self, bu_db):
        """Assign internal index values to each building unit."""
        for id, bu in enumerate(bu_db):
            # assign a temporary value for bu
            bu.internal_index = id
            for cp in bu.connect_points:
                cp.bu_order = id
        
    def exhaustive_sampling(self, bu_db):
        """
        Try every combination of bonds which are allowed in a
        tree-type generation where every new iteration over the
        structures should have the same number of building units
        in them.
        """
        # for timing purposes
        stopwatch = Time()
        stopwatch.timestamp()
        # create a local instance of bu_db
        base_building_units = self.expand_bu_list(bu_db)
        self.assign_internal_indices(base_building_units)
        # random seed 
        structures = self.random_insert(base_building_units)
        done = False
        # while loop
        while not done:
            add_list = []
            # keep track of symmetry labelling so redundant structures are
            # not made.
            symtrack = []
            for structure in structures:
                # scan over all building units
                building_units = structure.building_units
                # scan over all bonds with building units
                curr_bonds = [bond for bu in building_units
                         for bond in bu.connect_points]
                # scan over all library
                new_bonds = [bond for bu in base_building_units
                            for bond in bu.connect_points]
                # keep a list of all possible combinations
                bond_pairs = self.gen_bondlist(curr_bonds, new_bonds)
                for bond in bond_pairs:
                    # symmetry of bonds is determined to see if the bond has
                    # already been tried.
                    # grab the current building unit from the existing bond
                    # should be the first entry.
                    cbu = structure.building_units[bond[0].bu_order]
                    cbond = bond[0]
                    # the building unit to add is found in bond[1]
                    add_bu = base_building_units[bond[1].bu_order]
                    add_bond = bond[1]
                    # determine the symmetry of the bond trying to be formed
                    add_sym = self.get_symmetry_info(cbu, cbond, 
                                                     add_bu, add_bond)
                    # append the symmetry info to the growing structure.
                    new_id = structure.sym_id[:]
                    new_id.append(add_sym)
                    if tuple(new_id) not in symtrack:
                        # make a copy of the structure so it can be manipulated
                        # with out altering the existing structure.
                        new_struct = deepcopy(structure)
                        # re-reference the building units and connect_points
                        # using the self-referencing variables in bond[0]
                        cbu = new_struct.building_units[bond[0].bu_order]
                        cbond = cbu.connect_points[bond[0].order]
                        add_bu = deepcopy(base_building_units[bond[1].bu_order])
                        add_bond = add_bu.connect_points[bond[1].order]
                        #TODO(pboyd): this copying method really slows down the
                        #code, should find a better alternative.
                        debug("Added building unit #%i %s,"
                            %(len(structure.building_units),
                                add_bu.name)+
                            " bond %i, to building unit"
                            %(add_bond.index) +
                            " #%i  %s, bond %i"
                            %(cbu.order, cbu.name, cbond.index))
                        new_struct.insert(cbu, cbond, 
                                   add_bu, add_bond)
                        if not new_struct.overlap_bu(add_bu):
                            if (new_struct.saturated() and 
                                    new_struct.cell.index == 3 and 
                                    self.struct_building_set(new_struct,
                                                             base_building_units)):
                                stopwatch.timestamp()
                                info("Structure Generated! Timing reports "+
                                     "%f seconds"%stopwatch.timer)
                                
                                new_struct.get_scaled()
                                new_struct.finalize(base_building_units,
                                                    outdir = self.outdir,
                                                    csvfile = self.csv)
                                #fnl = Functionalize(new_struct, self.functional_groups)
                                #fnl.random_functionalization()
                                # NOTE the next two lines should be in a separate
                                # function which is or is not called based on
                                # a DEBUG flag.
                                file=open("debug.xyz", "w")
                                for addstr in add_list:
                                    file.writelines(addstr.xyz_lines)
                                file.close()
                                return
                            add_list.append(new_struct)
                            new_struct.sym_id = new_id[:]
                            # store symmetry data
                            symtrack.append(tuple(new_struct.sym_id[:])) 
                        else:
                            debug("overlap found")
                        if (len(add_list) + len(structures)) > 20000:
                            stopwatch.timestamp()
                            info("Genstruct went too long, "+
                            "%f seconds, returning..."%stopwatch.timer)
                            file=open("debug.xyz", "w")
                            for addstr in add_list:
                                file.writelines(addstr.xyz_lines)
                            file.close()
                            return
            if not add_list:
                stopwatch.timestamp()
                info("After %f seconds, "%stopwatch.timer +
                "no possible new combinations, returning...")
                file = open("debug.xyz", "w")
                for addstr in structures:
                    file.writelines(addstr.xyz_lines)
                file.close()
                return

            file=open("debug.xyz", "w")
            for addstr in add_list:
                file.writelines(addstr.xyz_lines)
            file.close()
            structures = add_list
        return

    def struct_building_set(self, struct, base_units):
        """Returns True if the structure contains representations of
        all the base_units.
        
        """
        base_indices = [bu.index for bu in base_units]
        base_indices = list(set(base_indices))
        base_indices.sort()
        
        struct_indices = [bu.index for bu in struct.building_units]
        struct_indices = list(set(struct_indices))
        struct_indices.sort()
        
        if base_indices == struct_indices:
            return True
        return False
        
    def get_symmetry_info(self, bu, cp, add_bu, add_cp):
        """Determine symmmetry, building unit types and distances between
        connect_points to make sure that no redundancy is done in the
        combinatorial growth of MOFs.

        """
        # determine the symmetry label
        symmetry_label = [(bu.internal_index, cp.symmetry), 
                          (add_bu.internal_index, add_cp.symmetry)]
        # check for if the bond is 'special'
        if (cp.special is not None) and (cp.special == add_cp.constraint):
            symmetry_label = [(bu.internal_index, "special%i"%(cp.special)),
                              (add_bu.internal_index,
                              "constraint%i"%(add_cp.constraint))]
        elif (cp.constraint is not None) and (cp.constraint == add_cp.special):
            symmetry_label = [(bu.internal_index, "constraint%i"%(cp.constraint)),
                              (add_bu.internal_index,
                              "special%i"%(add_cp.special))]
        symmetry_label.sort()
        symmetry_label = tuple(symmetry_label)
        return symmetry_label 


    def gen_bondlist(self, bonds, newbonds):
        """
        generate all possible combinations of bonds with newbonds.
        """
        bondlist = list(itertools.product(bonds, newbonds))
        pop = []
        for id, bondpair in enumerate(bondlist):
            # check for bond compatibility
            if not valid_bond(bondpair[0], bondpair[1]):
                pop.append(id)

        # remove illegal bonds from list
        pop.sort()
        [bondlist.pop(i) for i in reversed(pop)]
        return bondlist

    def random_insert(self, bu_db):
        """ selects a building unit at random to seed growth """
        bu = deepcopy(choice(bu_db))
        bu.order = 0
        seed = Structure()
        debug("Inserted %s as the initial seed for the structure"
                %(bu.name))
        seed.building_units.append(bu)
        # first entry in build directive is the index of the building unit.
        seed.directives.append((bu.index, bu.metal))
        for atom in bu.atoms:
            atom.index = seed.natoms
            seed.natoms += 1
        seed.store_building_unit_bonds(bu)
        # set building unit order = 0 for the connectpoints 
        for bu in seed.building_units:
            for bond in bu.connect_points:
                bond.bu_order = 0
        seed.debug_xyz()
        # check for initial bonding with itself (intermetal...)
        seed.bonding()      
        return [seed]

class Functionalize(object):
    """Class of methods to append functional groups to a Structure."""
    
    def __init__(self, structure, fg_database):
        self.max = 200  # Set upper limit of functionalizations per structure.
        self.structure = deepcopy(structure)
        self.functional_groups = deepcopy(fg_database)
        
    def get_base_units(self):
        """Returns a list of base building units from the structure."""
        dic = {}
        for building_unit in self.structure.building_units:
            # the dic keys are all the flags that make a
            # building unit unique.
            dic[(building_unit.index, building_unit.metal,
                 building_unit.parent)] = deepcopy(building_unit)
        # return the list of unique building_units
        return dic.values()

    def get_hydrogens(self, building_unit):
        """Returns a list of hydrogen atom indices in each building unit."""
        return [atom.internal_index for atom in
                building_unit.atoms if atom.element == "H"]
        
    def select(self, iterlist):
        """Returns a random selection for hydrogen atom
        atom substitution with a functional group.
        
        """
        # select a random number in the range of the list
        n = randrange(len(iterlist))
        # generate all possible combinations of h atoms
        pool = tuple(combinations(iterlist, n))
        # randomly choose one of these.
        return choice(pool)
    
    def choose_fnl_group(self, maxgroups):
        """Return a random selection of functional groups from a combination
        the length of which is determined by maxgroups.
        
        """
        rrange = range(len(self.functional_groups))
        pool = tuple(combinations(rrange, maxgroups))
        return [deepcopy(self.functional_groups[i]) for i in choice(pool)]
        
    def overlap(self, functional_group, bonded_atom, structure):
        """Determine if a particular functional group will overlap with the
        structure.  Returns a list of the functional group atoms which overlap
        with the structure.
        
        """
        sf = 0.4
        atom_list = []
        # periodic shift the atoms to the atoms in the functional group
        for atom in functional_group.atoms:
            mof_elements, mof_coords = structure.min_img_shift(
                                        atom=atom.coordinates[:3])
            # determine distances
            distances = distance.cdist([atom.coordinates[:3]], mof_coords)
            excl = bonded_atom.index
            for idx, dist in enumerate(distances[0]):
                if idx != excl:
                    max_dist = (Radii[atom.element] +
                                Radii[mof_elements[idx]]) * sf
                    if dist < max_dist:
                        atom_list.append(atom.index)
        return atom_list
        
    def random_functionalization(self, maxgroups=2):
        """Randomly functionalizes a structure. Max groups dictates
        the maximum number of functional groups included per Structure.
        
        What is randomized:
        1) the number and combination of hydrogen atoms on each building unit.
        2) the combination of functional groups for the MOF structure
        3) the assignment of a functional group to each hydrogen site.
        
        """
        h_atoms, functional_hist = {}, {}
        # determine base building units from structure
        base_building_units = self.get_base_units()
        # generate dictionary of hydrogen atoms for each building unit
        for building_unit in base_building_units:
            id = (building_unit.index, building_unit.metal,
                  building_unit.parent)
            hydrogen_sites = self.get_hydrogens(building_unit)
            h_atoms[id] = hydrogen_sites
        done = False
        while not done:
            # copy of structure to functionalize
            structure = deepcopy(self.structure)
            sites = {}
            # randomly select hydrogens on each unit
            for building_unit in base_building_units:
                id = (building_unit.index, building_unit.metal,
                      building_unit.parent)
                if h_atoms[id]:
                    sites[id] = self.select(h_atoms[id])
                else:
                    sites[id] = []
            # randomly choose functional groups
            assigned_fnlgrps = self.choose_fnl_group(maxgroups)
            # assign fnl groups to each hydrogen
            new_func = False
            while not new_func:
                record = []
                for key, value in sites.items():
                    # reset the site to a dictionary
                    # site[(building_unit)] = {hydrogen:functional_group}
                    sites[key] = {}
                    # descend into each hydrogen atom and assign a
                    # random choice of functional group.
                    h_record = []
                    for site in value:
                        group = choice(assigned_fnlgrps)
                        sites[key][site] = group
                        rec = tuple([site, group.index])
                        h_record.append(rec)
                    h_record.sort()
                    rec = tuple([key, tuple(h_record)])
                    record.append(rec)
                record.sort()
                # fastest check if the functional has been tried: dictionary
                # lookup?
                try:
                    functional_hist[tuple(record)]
                except KeyError:
                    # if the key is not in functional_hist, then exit this loop
                    new_func = True
            # keep record to prevent repeats.
            functional_hist[tuple(record)] = 1
            
            # iterate through each building unit in the structure, and
            # append functional groups
            for building_unit in structure.building_units:
                site_id = (building_unit.index, building_unit.metal,
                      building_unit.parent)
                replace_dic = sites[site_id]
                for hydrogen in replace_dic.keys():
                    fnl_grp = deepcopy(replace_dic[hydrogen])
                    H_atom = building_unit.atoms[hydrogen]
                    # find the associated bond
                    bond = building_unit.find_bonds(H_atom)
                    if len(bond) != 1:
                        raise Error("Hydrogen reports as having no bonds..")
                    bond = bond[0]
                    # nasty one-liner to assign the Joined atom
                    J_atom = bond.frm if bond.to == H_atom else bond.to
                    axis = bond.vector
                    # rotate and translate
                    fnl_grp.orient(bond.vector, J_atom.coordinates)
                    # check for overlap
                    overlap_atoms = self.overlap(fnl_grp, J_atom, structure)
                    if overlap_atoms:
                        # check if one of the overlap atoms is the
                        # connecting atom, in this case we just
                        # scrap the structure because no amount
                        # of rotation will help.
                        for overlap_atom in overlap_atoms:
                            if fnl_grp.atoms[overlap_atom].connectivity:
                                # need to break and try a different randomization.
                                # no amount of rotation will help here.
                                pass
                        # generate angles of 360 degrees split into 10 intervals.
                        rot_angle = 2 * pi / 10
                        tot_angle = 0.
                        while overlap_atoms and tot_angle < (2*pi):
                          fnl_grp.rotate(rot_angle)
                          tot_angle += rot_angle
                          overlap_atoms = self.overlap(fnl_grp, J_atom)
                    
                    # append the functional group to the Structure.
                    fnl_grp.MOF_atom = J_atom
                    building_unit.functionalize(fnl_grp)
            # update connectivities
            for building_unit in structure.building_units:
                pass
            # write cif file
            done = True 
    
    def symmetric_functionalization(self):
        """Symmetrically functionalizes a structure."""
        
        
class FunctionalGroupDatabase(list):
    """Reads in a file and stores the Functional Groups in a list."""
    
    def __init__(self, filename):
        self.readfile(filename)
        # get extension of file name without leading directories.
        self.extension = filename.split('/')[-1]
        self.extension = self.extension.split(".lib")[0]
            
    def readfile(self, filename):
        """
        Populate the list with building units from the
        input file.
        """
        file = ConfigParser.SafeConfigParser()
        file.read(filename)
        for idx, functional_group in enumerate(file.sections()):
            self.append(FunctionalGroup(
                                    name=functional_group,
                                    items=file.items(functional_group),
                                    index=idx))

class FunctionalGroup(object):
    """Defines a list of Atoms corresponding to a functional group."""
    
    def __init__(self, name, items, index=0):
        items = dict(items)
        self.index = index
        self.shname = name
        self.name = items.pop('name')
        self.atoms = []
        self.bonds = []
        # set up atoms and their connectivity
        self.build_atoms(items.pop('coordinates'), 
                          items.pop('table'))
        # centre of mass
#        self.calculate_COM()
        self.normal = np.array([float(i) for i in items.pop('normal').split()])
        self.connect_vector = np.ones(4)
        self.connect_vector[:3] = np.array([float(i) for i in
                                            items.pop('orientation').split()])
        self.bond_length = float(items.pop('carbon_bond'))
        self.MOF_bond = None  # this will be the MOFs carbon atom it bonds to
        
    def build_atoms(self, coordinates, table):
        ind = 0
        for atom in coordinates.splitlines():
            atom = atom.strip()
            if not atom:
                continue
            self.atoms.append(Atom_gen(atom))
            self.atoms[-1].index = ind
            self.atoms[-1].bu_index = None
            self.atoms[-1].bu_metal = False
            self.atoms[-1].fnl_group_index = self.index
            ind += 1
        table = table.strip()
        # self.connect_points must be populated before doing this.
        for bond in table.splitlines():
            bond = bond.strip().split()
            # add the bonding information
            # first two cases are for bonding to connecting points
            if "c" in bond[0].lower():
                atom_ind = int(bond[1])
                self.atoms[atom_ind].connectivity = 1

            elif "c" in bond[1].lower():
                # subtract 1 since the input file starts at 1
                atom_ind = int(bond[0])
                self.atoms[atom_ind].connectivity = 1
            else:
                # add bonding to atoms
                # add the index of the atom it is bonded with
                self.atoms[int(bond[0])].bonds.append(
                        int(bond[1]))
                self.atoms[int(bond[1])].bonds.append(
                        int(bond[0]))
                self.bonds.append(Bond(
                     self.atoms[int(bond[0])], 
                     self.atoms[int(bond[1])], bond[2]))
                
    def orient(self, vector, point):
        """Shift the functional groups' atoms and vectors to align with
        the vector.
        
        NOTE: this is not tested, needs to be tested.
        
        """
        vector = unit_vector(vector)
        angle = calc_angle(self.connect_vector[:3], vector[:3])
        axis = np.cross(self.connect_vector[:3], vector[:3])
        pt = [i.index for i in self.atoms if i.connectivity]

        # pt is the functional group atom which bonds to the structure
        pt = pt[0]
        R = rotation_matrix(axis, angle, point=self.atoms[pt].coordinates[:3])
        # NONE of this rotation stuff has been tested yet.
        for atom in self.atoms:
            atom.coordinates = np.dot(R, atom.coordinates)
        self.normal = np.dot(R[:3,:3], self.normal)
        self.connect_vector = np.dot(R[:3,:3], self.connect_vector[:3])
        trans_v = (self.atoms[pt].coordinates[:3] - point[:3] +
                    self.bond_length * vector[:3])
        for atom in self.atoms:
            atom.coordinates[:3] = atom.coordinates[:3] + trans_v
            
    def rotate(self, angle):
        """Rotate a functional group about it's axis with the angle supplied."""
        axis = self.connect_vector
        # point about which to rotate: the connecting atom
        pt = [i.index for i in self.atoms if i.connectivity]
        pt = pt[0]               
        R = rotation_matrix(axis, angle, point=self.atoms[pt].coordinates)
        # rotate the atom coordinates
        for atom in self.atoms:
            atom.coordinates = np.dot(R, atom.coordinates)        
        # rotate the normal, though I don't think this matters
        self.normal[:3] = np.dot(R[:3,:3], self.normal[:3])
        

def valid_bond(bond, newbond):
    """
    Determines if two connect_points are compatible
    1) checks to see if both have the same special/constraint flag, which 
        should be an [int] or [None]
    2) if [None] then make sure both are not metal bonding units
    
    otherwise return False.
    """
    # check if the bond is already bonded
    if bond.bonded or newbond.bonded:
        return False
    # check if one of the bonds is special.  Then make sure the other bond
    # is constrained to that special bond.
    if bond.special is not None:
        if newbond.constraint == bond.special:
            return True
    if bond.constraint is not None:
        if newbond.special == bond.constraint:
            return True
    # check if the bond is metal-metal
    if (newbond.special is None)and(newbond.constraint is None)and(
        bond.special is None)and(bond.constraint is None):
        if bond.metal != newbond.metal:
            return True
    # otherwise return False
    return False

def xyz_file(atoms=None, coordinates=None, vectors=None):
    """Writes an xyz file with vectors if specified"""
    natoms = len(atoms)
    atoms[0] = "Rn"
    xyzstream = open("xyz.xyz", "a")
    atmline = "%-5s%12.5f%12.5f%12.5f\n"
    vectorline = "%-5s%12.5f%12.5f%12.5f atom_vector %12.5f%12.5f%12.5f"
    xyzstream.writelines("%i\n%s\n"%(natoms, "xyz"))
    for atom, (x, y, z) in zip(atoms, coordinates):
        line = atmline%(atom,x,y,z)
        xyzstream.writelines(line)
    xyzstream.close()

def main():
    if len(sys.argv) > 1:
        file = sys.argv[1]
    else:
        file = "testdb.dat"
    data = Database(file)
    base = file.split('/')[-1]
    base = base.split(".dat")[0]
    Log(file=base + ".out")
    Generate(data)

if __name__ == '__main__':
    main()
