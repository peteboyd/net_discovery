#!/usr/bin/env python
import sys
import io
from ConfigParser import ConfigParser
from scipy.spatial import distance
from operations import *

class Functional_groups(object):
    """
    Functional groups contain connectivity info, file reading etc.
    """
    def __init__(self):
        """
        Read in library of functional groups, build arrays.
        """
        lib_file = "functional_groups.lib"
        #lib_file = "group.lib"
        self.groups = ConfigParser()
        self.groups.read(lib_file)
        self.name = {}
        self.atoms = {}
        self.atoms_fftype = {}
        self.coordinates = {}
        self.connect_vector = {}
        self.connect_align = {}
        self.bond_length = {}
        self.connect_points = {}
        self.table_exists = False
        self.bondspec = {}
        self.table = {}
        self.populate_arrays()

    def populate_arrays(self):
        """
        Populate the self.atoms, self.coordinates, self.connect_vectors
        arrays.
        """
        fnl_groups = self.groups.sections()
        for group in fnl_groups:
            self.bondspec = {}
            lines = io.BytesIO(self.groups.get(group, "atoms").\
                    strip("\n")).readlines()
            tmpatoms = []
            tmpfftype = []
            tmpcoord = []
            for line in lines:
                line = line.strip("\n")
                tmpatoms.append(line.split()[0])
                tmpfftype.append(line.split()[1])
                tmpcoord.append([float(i) for i in line.split()[2:5]])

            if self.groups.has_option(group, "table"):
                self.table_exists = True
                lines = io.BytesIO(self.groups.get(group, "table").\
                        strip("\n")).readlines()
                for line in lines:
                    line = line.strip("\n")
                    idx1 = int(line.split()[0])
                    idx2 = int(line.split()[1])
                    type = line.split()[2]

                    self.bondspec.setdefault(idx1, []).\
                            append((idx2, type))
                    self.bondspec.setdefault(idx2, []).\
                            append((idx1, type))

            idx = self.groups.getint(group,"index")
            self.atoms[idx] = tmpatoms[:]
            self.coordinates[idx] = tmpcoord[:]
            self.atoms_fftype[idx] = tmpfftype[:]
            self.connect_vector[idx] = [float(i) for i in 
                    self.groups.get(group, "orientation").split()]
            self.bond_length[idx] = self.groups.getfloat(group, "carbon_bond")
            self.connect_align[idx] = [float(i) for i in 
                    self.groups.get(group, "normal").split()]
            self.connect_points[idx] = self.groups.getint(group, "connection_point")
            self.name[idx] = self.groups.get(group, "name")
            self.coord_matrix(idx)

    def coord_matrix(self, idx):
        """
        Generate a coordination matrix for the SBU's coordinates.
        """
        numatms = len(self.atoms[idx])
        # populate empty connectivity table
        self.table[idx] = [[0.] * numatms for i in xrange(numatms)]

        distmatrx = distance.cdist(self.coordinates[idx], 
                                   self.coordinates[idx])
        if self.table_exists:
            for i in self.bondspec.keys():
                for j in self.bondspec[i]:
                    self.table[idx][i][j[0]] = (distmatrx[i][j[0]],
                                                j[1])
        else:
            for i in range(numatms):
                for j in range(i+1, numatms):
                    tol = bond_tolerance(self.atoms[idx][i], 
                                     self.atoms[idx][j])
                    if (i != j) and (distmatrx[i,j] <= tol):
                        self.table[idx][i][j] = (distmatrx[i,j], "S")
                        self.table[idx][j][i] = (distmatrx[j,i], "S")
        return 

        
def main():
    fnlgrp = Functional_groups()
    print fnlgrp.table
if __name__ == "__main__":
    main()

