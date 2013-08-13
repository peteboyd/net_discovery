#!/usr/bin/env python
import options
import sys
import os
options = options.Options()
sys.path.append(options.faps_dir)
from faps import Structure, Cell, Atom, Symmetry
from function_switch import FunctionalGroupLibrary, FunctionalGroup
from elements import CCDC_BOND_ORDERS
sys.path.append(options.genstruct_dir)
from genstruct import Database, BuildingUnit, Atom_gen
from logging import info, debug, warning, error, critical

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

    def _line_count(self, filename):
        with open(filename) as f:
            for i, l in enumerate(f):
                pass
        return i + 1

    def get(self, column):
        assert column in self._data.keys()
        return self._data[column]

def main():
    mofs = CSV(options.csv_file)

    for mof in mofs.get('MOFname'):
        print mof
        cif = Structure(mof)
        try:
            cif.from_file(os.path.join(options.lookup,
                         mof), "cif", '')
        except IOError:
            cif.from_file(os.path.join(options.lookup,
                          mof+".out"), "cif", '')


if __name__=="__main__":
    main()
