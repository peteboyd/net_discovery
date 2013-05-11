#!/usr/bin/env python

"""
reads and writes files, sets up logging
"""

import logging
import sys
import textwrap
from copy import copy, deepcopy
from datetime import date
from time import time
from logging import warning, debug, error, info, critical
from operations import *
from elements import *
import numpy as np
try:
    import pyspglib._spglib as spg
except ImportError:
    pass
    #warning("Symmetry library couldn't be loaded, all structures will "+
    #        "have P1 symmetry.")

xyzbondfmt = "%s%12.5f%12.5f%12.5f " +\
             "atom_vector%12.5f%12.5f%12.5f " +\
             "atom_vector%12.5f%12.5f%12.5f\n"
xyzcellfmt1 = "%s%12.5f%12.5f%12.5f " +\
             "atom_vector%12.5f%12.5f%12.5f\n"
xyzcellfmt2 = "%s%12.5f%12.5f%12.5f " +\
             "atom_vector%12.5f%12.5f%12.5f " +\
             "atom_vector%12.5f%12.5f%12.5f " +\
             "atom_vector%12.5f%12.5f%12.5f\n"
xyzatomfmt = "%s%12.5f%12.5f%12.5f\n"

#pdb format determined from description in pdb document
pdbatmfmt = "%-6s%5i%5s%1s%3s%2s%4i%1s%11.3f%8.3f%8.3f%6.2f%6.2f%12s%2s\n"
pdbcellfmt = "%-6s%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f P1\n"

class Multidict(dict):
    """ change keys for reading input with ConfigParser """
    _unique = ""
    def __setitem__(self, key, val):
        if isinstance(val, dict):
            self._unique += "_i"
            key += str(self._unique)
        dict.__setitem__(self, key, val)

class Log(object):

    def __init__(self, file="genstruct.out"):
        self.file = file
        self.quiet = False 
        self.verbose = True
        self.default = False
        self._init_logging()
        # set up writing to file and terminal

    def _init_logging(self):
        if self.quiet:
            stdout_level = logging.ERROR
            file_level = logging.INFO
        elif self.verbose:
            stdout_level = logging.DEBUG
            file_level = logging.DEBUG
        elif self.default:
            stdout_level = logging.INFO
            file_level = logging.DEBUG
        else:
            stdout_level = logging.INFO
            file_level = logging.INFO           

        logging.basicConfig(level=file_level,
                            format='[%(asctime)s] %(levelname)s %(message)s',
                            datefmt='%Y%m%d %H:%M:%S',
                            filename=self.file,
                            filemode='a')
       
        logging.addLevelName(10, '--')
        logging.addLevelName(20, '>>')
        logging.addLevelName(30, '**')
        logging.addLevelName(40, '!!')
        logging.addLevelName(50, 'XX')

        console = ColouredConsoleHandler(sys.stdout)
        console.setLevel(stdout_level)
        console.setFormatter(logging.Formatter('%(levelname)s %(message)s'))
        logging.getLogger('').addHandler(console)

class ColouredConsoleHandler(logging.StreamHandler):
    """Makes colourised and wrapped output for the console."""
    def emit(self, record):
        """Colourise and emit a record."""
        # Need to make a actual copy of the record
        # to prevent altering the message for other loggers
        myrecord = copy(record)
        levelno = myrecord.levelno
        if levelno >= 50:  # CRITICAL / FATAL
            front = '\033[30;41m'  # black/red
            text = '\033[30;41m'  # black/red
        elif levelno >= 40:  # ERROR
            front = '\033[30;41m'  # black/red
            text = '\033[1;31m'  # bright red
        elif levelno >= 30:  # WARNING
            front = '\033[30;43m'  # black/yellow
            text = '\033[1;33m'  # bright yellow
        elif levelno >= 20:  # INFO
            front = '\033[30;42m'  # black/green
            text = '\033[1m'  # bright
        elif levelno >= 10:  # DEBUG
            front = '\033[30;46m'  # black/cyan
            text = '\033[0m'  # normal
        else:  # NOTSET and anything else
            front = '\033[0m'  # normal
            text = '\033[0m'  # normal

        myrecord.levelname = '%s%s\033[0m' % (front, myrecord.levelname)
        myrecord.msg = textwrap.fill(
            myrecord.msg, initial_indent=text, width=76,
            subsequent_indent='\033[0m   %s' % text) + '\033[0m'
        logging.StreamHandler.emit(self, myrecord)

class CSV(object):
    """
    writes a .csv file with data for the run.
    """

    def __init__(self):
        self.data = []
        self._csvname = "default"
        self._datalen = 0 

    def add_data(self, **kwargs):
        # make sure the length of each entry in data is the same.
        self.data.append(kwargs)
        self._datalen += 1

    def set_name(self, name):
        self._csvname = name

    def write_file(self):
        # Designed the CSV file specifically to report the titles in the order 
        # here.
        titles = "#MOFname,metal_index,organic_index1,organic_index2,"+\
                "H_M_symmetry_name,symmetry_number,build_directive\n"
        data = ""
        for entry in range(self._datalen):
            data += "%(MOFname)s,%(metal_index)i,"%(self.data[entry])
            data += "%(organic_index1)i,%(organic_index2)i,"%(self.data[entry])
            data += "%(h_m_symmetry_name)s,%(symmetry_number)i,"%(self.data[entry])
            data += "%(build_directive)s\n"%(self.data[entry])
        lines = titles + data
        csvfile = open(self._csvname + '.csv', "w")
        csvfile.writelines(lines)
        csvfile.close()
        
class CIF(object):
    """
    Write cif files
    """
    def __init__(self, struct, sym=True, tol=0.4):
        """Store the structure class and apply symmetry to the system
        the symmetry will be dictated by the above values of sym and tol.
        
        """
        self.struct = struct
        self.symmetry = Symmetry(struct, sym=sym, symprec=tol)
        
    def add_labels(self, equiv_atoms):
        """
        Include numbered labels for each atom.
        """
        atomdic = {}
        labeldic = {}
        for unq_atom in set(equiv_atoms):
            label = self.symmetry._element_symbols[unq_atom]
            atomdic.setdefault(label,0)
            atomdic[label] += 1
            labeldic[unq_atom] = label + str(atomdic[label])

        label = []
        for atom in equiv_atoms:
            label.append(labeldic[atom])

        return label 

    def write_cif(self, name=None):
        """Writes the cif file based on the symmetry found and the bonding data
        stored in the structure.
        
        """

        method = "Genstruct - created by [your name here]"
        if name is None:
            filename = "default.cif"
        else:
            filename = name + ".cif"
        # determine cell parameters
        cell = self.symmetry._lattice.copy()
        cellparams = self.get_cell_params(cell)
        #lines = "data_" + str(name).split('/')[1] + "\n"
        lines = "data_" + str(name) + "\n"
        today = date.today()
        prefix = "_audit"
        # creation date (under _audit)
        lines += "%--34s"%(prefix + "_creation_date") + \
                today.strftime("%A %d %B %Y") + "\n"
        # creation method (under _audit)
        lines += "%-34s"%(prefix + "_creation_method") + \
                method + "\n\n"

        prefix = "_symmetry"
        space_group_name = self.symmetry.get_space_group_name()
        # space group name (under _symmetry)
        lines += "%-34s"%(prefix + "_space_group_name_H-M") + \
                space_group_name + "\n"
       
        space_group_number = self.symmetry.get_space_group_number()
        # space group number (under _symmetry)
        lines += "%-34s"%(prefix + "_Int_Tables_number") + \
                str(space_group_number) + "\n"
        # cell setting (under _symmetry)

        if space_group_number == 0:
            print "space group name", space_group_name
        lines += "%-34s"%(prefix + "_cell_setting") + \
                cell_setting[space_group_number] + "\n"

        lines += "\n"
        # symmetry equivalent positions
        lines += "loop_\n"
        lines += prefix + "_equiv_pos_as_xyz\n"
        sym_ops = self.symmetry.get_space_group_operations()

        for op in sym_ops:
            lines += "'%s'\n"%op
        #for op in SYM_OPS[space_group_name]:
        #    lines += "'%s, %s, %s'\n"%tuple([i for i in op.split(",")])

        lines += "\n"
        prefix = "_cell"
        # cell parameters (under _cell)
        lines += "%-34s"%(prefix + "_length_a") + \
                "%(a)-7.4f\n"%(cellparams)
        lines += "%-34s"%(prefix + "_length_b") + \
                "%(b)-7.4f\n"%(cellparams)
        lines += "%-34s"%(prefix + "_length_c") + \
                "%(c)-7.4f\n"%(cellparams)
        lines += "%-34s"%(prefix + "_angle_alpha") + \
                "%(alpha)-7.4f\n"%(cellparams)
        lines += "%-34s"%(prefix + "_angle_beta") + \
                "%(beta)-7.4f\n"%(cellparams)
        lines += "%-34s"%(prefix + "_angle_gamma") + \
                "%(gamma)-7.4f\n\n"%(cellparams)
        # fractional coordinates
        lines += "loop_\n"

        lines += "_atom_site_label\n"
        lines += "_atom_site_type_symbol\n"
        lines += "_atom_site_description\n"
        lines += "_atom_site_fract_x\n"
        lines += "_atom_site_fract_y\n"
        lines += "_atom_site_fract_z\n"

        # now check to see which atoms to keep in the cif file.
        equivalent_atoms = self.symmetry.get_equiv_atoms()
        unique_atoms = list(set(equivalent_atoms)) 
        unique_coords = [list(self.symmetry._scaled_coords[i]) for i in unique_atoms]
        unique_symbols = [self.symmetry._element_symbols[i] for i in unique_atoms]
        unique_labels = self.add_labels(equivalent_atoms)
        atoms_fftype = [atom.force_field_type for bu in 
                self.struct.building_units for atom in bu.atoms]
        len_orig = self.struct.natoms
        # not sure if total_fftype is necessary.  I think that symmetry finding
        # reports all unique atoms from the original set, not the extended set
        total_fftype = [atoms_fftype[i%len_orig] for i in range(
            len(self.symmetry._scaled_coords))]
        unique_fftype = [total_fftype[i] for i in unique_atoms]
        for idx, atom in enumerate(unique_atoms):
            line = [unique_labels[atom], unique_symbols[idx], 
                    unique_fftype[idx]] + unique_coords[idx]
            lines += "%-7s%-6s%-5s%10.5f%10.5f%10.5f\n"%(tuple(line))
        lines += "\n"
        connect_table = self.update_connect_table(equivalent_atoms)
        lines += "loop_\n"
        lines += "_geom_bond_atom_site_label_1\n"
        lines += "_geom_bond_atom_site_label_2\n"
        lines += "_geom_bond_distance\n"
        lines += "_ccdc_geom_bond_type\n"
        for bond in connect_table:
            lines += "%-7s%-7s%10.5f%5s\n"%(unique_labels[bond[0]],
                unique_labels[bond[1]], 
                bond[3], bond[2]) 

        ciffile = open(filename, 'w')
        ciffile.writelines(lines)
        ciffile.close()
        return

    def update_connect_table(self, equivalent_atoms):
        """Return only the bonds which exist between asymmetric atoms."""
        conn_table = []
        equiv_atoms = {}
        for idx, atom in enumerate(equivalent_atoms):
            equiv_atoms[idx] = atom
        tempdic = {}
        atomlist = [i for bu in self.struct.building_units for i in bu.atoms]
        for atm1, atm2, type, length in self.struct.bonds:
            eatm1 = equiv_atoms[atm1]
            eatm2 = equiv_atoms[atm2]
            atmind = [eatm1, eatm2]
            atmind.sort()
            tempdic[tuple(atmind)] = (eatm1, eatm2, type, length)
        return tempdic.values() 

    def index_check(self, index, uniques):
        """ 
        check if the index is already in uniques, or it's equivalent 
        is in uniques.
        """
        if index in uniques:
            return False
        for i in self.equiv_dic[index]:
            if i in uniques:
                return False
        return True

    def get_cell_params(self, cell):
        """ return alen, blen, clen, alpha, beta, gamma """

        cellparams = {}
        cellparams['a'] = length(cell[0])
        cellparams['b'] = length(cell[1])
        cellparams['c'] = length(cell[2])
        cellparams['alpha'] = calc_angle(cell[1], cell[2])*RAD2DEG
        cellparams['beta'] = calc_angle(cell[0], cell[2])*RAD2DEG
        cellparams['gamma'] = calc_angle(cell[0], cell[1])*RAD2DEG
        return cellparams


class Symmetry(object):
    """
    Symmetry class to calculate the symmetry elements of a given MOF
    """
    def __init__(self, structure, sym=True, symprec=1.e-5):

        self.sym = sym
        self.struct = structure
        self._symprec = symprec
        self._lattice = structure.cell.lattice.copy()
        self._scaled_coords = np.array([atom.scaled_coords for bu in 
            structure.building_units for atom in bu.atoms])
        #_angle_tol represents the tolerance of angle between lattice
        # vectors in degrees.  Negative value invokes converter from
        # symprec
        self._angle_tol = -1.0
        self._element_symbols = [atom.element for bu in 
                structure.building_units for atom in bu.atoms]
        self._numbers = np.array([ATOMIC_NUMBER.index(i) for i in 
                                  self._element_symbols])
        self.dataset = {}
        if self.sym:
            try:
                spg
            except NameError:
                self.sym = False
        self.refine_cell()

    def refine_cell(self):
        """
        get refined data from symmetry finding
        """
        if self.sym:
            # Temporary storage of structure info
            _lattice = self._lattice.T.copy()
            _scaled_coords = self._scaled_coords.copy()
            _symprec = self._symprec
            _angle_tol = self._angle_tol
            _numbers = self._numbers.copy()
            
            keys = ('number',
                    'international',
                    'hall',
                    'transformation_matrix',
                    'origin_shift',
                    'rotations',
                    'translations',
                    'wyckoffs',
                    'equivalent_atoms')
            dataset = {}

            dataset['number'] = 0
            while dataset['number'] == 0:

                # refine cell
                num_atom = len(_scaled_coords)
                ref_lattice = _lattice.copy()
                ref_pos = np.zeros((num_atom * 4, 3), dtype=float)
                ref_pos[:num_atom] = _scaled_coords.copy()
                ref_numbers = np.zeros(num_atom * 4, dtype=int)
                ref_numbers[:num_atom] = _numbers.copy()
                num_atom_bravais = spg.refine_cell(ref_lattice,
                                           ref_pos,
                                           ref_numbers,
                                           num_atom,
                                           _symprec,
                                           _angle_tol)
                for key, data in zip(keys, spg.dataset(ref_lattice.copy(),
                                        ref_pos[:num_atom_bravais].copy(),
                                    ref_numbers[:num_atom_bravais].copy(),
                                                _symprec,
                                                _angle_tol)):
                    dataset[key] = data

                _symprec = _symprec * 0.5

            # an error occured with met9, org1, org9 whereby no
            # symmetry info was being printed for some reason.
            # thus a check is done after refining the structure.

            if dataset['number'] == 0:
                warning("WARNING - Bad Symmetry found!")
                self.sym = False
            else:

                self.dataset['number'] = dataset['number']
                self.dataset['international'] = dataset['international'].strip()
                self.dataset['hall'] = dataset['hall'].strip()
                self.dataset['transformation_matrix'] = np.array(dataset['transformation_matrix'])
                self.dataset['origin_shift'] = np.array(dataset['origin_shift'])
                self.dataset['rotations'] = np.array(dataset['rotations'])
                self.dataset['translations'] = np.array(dataset['translations'])
                letters = "abcdefghijklmnopqrstuvwxyz"
                self.dataset['wyckoffs'] = [letters[x] for x in dataset['wyckoffs']]
                self.dataset['equivalent_atoms'] = np.array(dataset['equivalent_atoms'])
                self._lattice = ref_lattice.T.copy()
                self._scaled_coords = ref_pos[:num_atom_bravais].copy()
                self._numbers = ref_numbers[:num_atom_bravais].copy()
                self._element_symbols = [ATOMIC_NUMBER[i] for 
                    i in ref_numbers[:num_atom_bravais]]

    def get_space_group_name(self):

        if self.sym:
            #TEMP: pretend "P1"
            #return "P1"
            return self.dataset["international"] 
        else:
            return "P1"

    def get_space_group_operations(self):

        if self.sym:
            return [self.convert_to_string((r, t)) 
                    for r, t in zip(self.dataset['rotations'], 
                                    self.dataset['translations'])]
        else:
            # P1
            return ["x, y, z"]

    def convert_to_string(self, operation):
        """ takes a rotation matrix and translation vector and
        converts it to string of the format "x, y, z" """
       
        # operation[0][0] is the first entry,
        # operation[0][1] is the second entry,
        # operation[0][2] is the third entry,
        # operation[1][1, 2, 3] are the translations
        fracs = [tofrac(i) for i in operation[1]]
        string = ""
        for idx, op in enumerate(operation[0]):
            x,y,z = op
            # note, this assumes 1, 0 entries in the rotation operation
            str_conv = (to_x(x), "+"*abs(x)*(y) + to_y(y),
                        "+"*max(abs(x),abs(y))*(z) + to_z(z))
            # determine if translation needs to be included
            for ind, val in enumerate((x,y,z)):
                frac = ""
                if val and fracs[ind][1] != 0:
                    frac = "+%i/%i"%(fracs[ind][1], fracs[ind][2])
                string += str_conv[ind] + frac

            # function to add comma delimiter if not the last entry
            f = lambda p: p < 2 and ", " or ""
            string += f(idx) 

        return string

    def get_space_group_number(self):

        if self.sym:
            #TEMP: pretend "P1"
            #return 1
            return self.dataset["number"]
        else:
            return 1

    def get_equiv_atoms(self):
        """Returs a list where each entry represents the index to the
        asymmetric atom. If P1 is assumed, then it just returns a list
        of the range of the atoms."""
        if self.sym:
            return self.dataset["equivalent_atoms"]
        else:
            return range(len(self._element_symbols))


class Time:
    """
    Class to time executions
    """
    def __init__(self):
        self.timer = 0.
        self.currtime = time() 

    def timestamp(self):
        currtime = time()
        self.timer = currtime - self.currtime 
        self.currtime = currtime

def writexyz(atoms, coordinates, name="default"):
    """ Writes a general xyz coordinate file for viewing """

    name = name + ".xyz"
    file = open(name, "w")
    file.writelines("%5i\n%s\n"%(len(atoms), name))
    for atom, coord in zip(atoms, coordinates):
        file.writelines(xyzatomfmt%tuple([atom] + list(coord))) 

    file.close()
    return

cell_setting = {
    0   :   "triclinic",
    1   :   "triclinic",       
    2   :   "triclinic",       
    3   :   "monoclinic",      
    4   :   "monoclinic",      
    5   :   "monoclinic",      
    6   :   "monoclinic",      
    7   :   "monoclinic",      
    8   :   "monoclinic",      
    9   :   "monoclinic",      
   10   :   "monoclinic",      
   11   :   "monoclinic",      
   12   :   "monoclinic",      
   13   :   "monoclinic",      
   14   :   "monoclinic",      
   15   :   "monoclinic",      
   16   :   "orthorhombic",      
   17   :   "orthorhombic",      
   18   :   "orthorhombic",      
   19   :   "orthorhombic",      
   20   :   "orthorhombic",      
   21   :   "orthorhombic",      
   22   :   "orthorhombic",      
   23   :   "orthorhombic",      
   24   :   "orthorhombic",      
   25   :   "orthorhombic",      
   26   :   "orthorhombic",      
   27   :   "orthorhombic",      
   28   :   "orthorhombic",      
   29   :   "orthorhombic",      
   30   :   "orthorhombic",      
   31   :   "orthorhombic",      
   32   :   "orthorhombic",      
   33   :   "orthorhombic",      
   34   :   "orthorhombic",      
   35   :   "orthorhombic",      
   36   :   "orthorhombic",      
   37   :   "orthorhombic",      
   38   :   "orthorhombic",      
   39   :   "orthorhombic",      
   40   :   "orthorhombic",      
   41   :   "orthorhombic",      
   42   :   "orthorhombic",      
   43   :   "orthorhombic",      
   44   :   "orthorhombic",      
   45   :   "orthorhombic",      
   46   :   "orthorhombic",      
   47   :   "orthorhombic",      
   48   :   "orthorhombic",      
   49   :   "orthorhombic",      
   50   :   "orthorhombic",      
   51   :   "orthorhombic",      
   52   :   "orthorhombic",      
   53   :   "orthorhombic",      
   54   :   "orthorhombic",      
   55   :   "orthorhombic",      
   56   :   "orthorhombic",      
   57   :   "orthorhombic",      
   58   :   "orthorhombic",      
   59   :   "orthorhombic",      
   60   :   "orthorhombic",      
   61   :   "orthorhombic",      
   62   :   "orthorhombic",      
   63   :   "orthorhombic",      
   64   :   "orthorhombic",      
   65   :   "orthorhombic",      
   66   :   "orthorhombic",      
   67   :   "orthorhombic",      
   68   :   "orthorhombic",      
   69   :   "orthorhombic",      
   70   :   "orthorhombic",      
   71   :   "orthorhombic",      
   72   :   "orthorhombic",      
   73   :   "orthorhombic",      
   74   :   "orthorhombic",      
   75   :   "tetragonal",        
   76   :   "tetragonal",        
   77   :   "tetragonal",        
   78   :   "tetragonal",        
   79   :   "tetragonal",        
   80   :   "tetragonal",        
   81   :   "tetragonal",        
   82   :   "tetragonal",        
   83   :   "tetragonal",        
   84   :   "tetragonal",        
   85   :   "tetragonal",        
   86   :   "tetragonal",        
   86   :   "tetragonal",        
   87   :   "tetragonal",        
   88   :   "tetragonal",        
   89   :   "tetragonal",        
   90   :   "tetragonal",        
   91   :   "tetragonal",        
   92   :   "tetragonal",        
   93   :   "tetragonal",        
   94   :   "tetragonal",        
   95   :   "tetragonal",        
   96   :   "tetragonal",        
   97   :   "tetragonal",        
   98   :   "tetragonal",        
   99   :   "tetragonal",        
  100   :   "tetragonal",        
  101   :   "tetragonal",        
  102   :   "tetragonal",        
  103   :   "tetragonal",        
  104   :   "tetragonal",        
  105   :   "tetragonal",        
  106   :   "tetragonal",        
  107   :   "tetragonal",        
  108   :   "tetragonal",        
  109   :   "tetragonal",        
  110   :   "tetragonal",        
  111   :   "tetragonal",        
  112   :   "tetragonal",        
  113   :   "tetragonal",        
  114   :   "tetragonal",        
  115   :   "tetragonal",        
  116   :   "tetragonal",        
  117   :   "tetragonal",        
  118   :   "tetragonal",        
  119   :   "tetragonal",        
  120   :   "tetragonal",        
  121   :   "tetragonal",        
  122   :   "tetragonal",        
  123   :   "tetragonal",        
  124   :   "tetragonal",        
  125   :   "tetragonal",        
  126   :   "tetragonal",        
  127   :   "tetragonal",        
  128   :   "tetragonal",        
  129   :   "tetragonal",        
  130   :   "tetragonal",        
  131   :   "tetragonal",        
  132   :   "tetragonal",        
  133   :   "tetragonal",        
  134   :   "tetragonal",        
  135   :   "tetragonal",        
  136   :   "tetragonal",        
  137   :   "tetragonal",        
  138   :   "tetragonal",        
  139   :   "tetragonal",        
  140   :   "tetragonal",        
  141   :   "tetragonal",        
  142   :   "tetragonal",        
  143   :   "trigonal",          
  144   :   "trigonal",          
  145   :   "trigonal",          
  146   :   "rhombohedral",   
  147   :   "trigonal",       
  148   :   "rhombohedral",   
  149   :   "trigonal",       
  150   :   "trigonal",       
  151   :   "trigonal",       
  152   :   "trigonal",       
  153   :   "trigonal",       
  154   :   "trigonal",       
  155   :   "rhombohedral",   
  156   :   "trigonal",       
  157   :   "trigonal",       
  158   :   "trigonal",       
  159   :   "trigonal",       
  160   :   "rhombohedral",   
  161   :   "rhombohedral",   
  162   :   "trigonal",       
  163   :   "trigonal",       
  164   :   "trigonal",       
  165   :   "trigonal",       
  166   :   "rhombohedral",   
  167   :   "rhombohedral",   
  168   :   "hexagonal",      
  169   :   "hexagonal",      
  170   :   "hexagonal",      
  171   :   "hexagonal",      
  172   :   "hexagonal",      
  173   :   "hexagonal",      
  174   :   "hexagonal",      
  175   :   "hexagonal",      
  176   :   "hexagonal",      
  177   :   "hexagonal",      
  178   :   "hexagonal",      
  179   :   "hexagonal",      
  180   :   "hexagonal",      
  181   :   "hexagonal",      
  182   :   "hexagonal",      
  183   :   "hexagonal",      
  184   :   "hexagonal",      
  185   :   "hexagonal",      
  186   :   "hexagonal",      
  187   :   "hexagonal",      
  188   :   "hexagonal",      
  189   :   "hexagonal",      
  190   :   "hexagonal",      
  191   :   "hexagonal",      
  192   :   "hexagonal",      
  193   :   "hexagonal",      
  194   :   "hexagonal",      
  195   :   "cubic",          
  196   :   "cubic",          
  197   :   "cubic",          
  198   :   "cubic",          
  199   :   "cubic",          
  200   :   "cubic",          
  201   :   "cubic",          
  202   :   "cubic",          
  203   :   "cubic",          
  204   :   "cubic",          
  205   :   "cubic",          
  206   :   "cubic",          
  207   :   "cubic",          
  208   :   "cubic",          
  209   :   "cubic",          
  210   :   "cubic",          
  211   :   "cubic",          
  212   :   "cubic",          
  213   :   "cubic",          
  214   :   "cubic",          
  215   :   "cubic",          
  216   :   "cubic",          
  217   :   "cubic",          
  218   :   "cubic",          
  219   :   "cubic",          
  220   :   "cubic",          
  221   :   "cubic",          
  222   :   "cubic",          
  223   :   "cubic",          
  224   :   "cubic",          
  225   :   "cubic",          
  226   :   "cubic",          
  227   :   "cubic",          
  228   :   "cubic",          
  229   :   "cubic",          
  230   :   "cubic",          
  } 

# main group values taken from J.Phys.Chem.A, 113, 5811, 2009
# transition metal radii taken from Bondi JPC '64
#Radii = {
#  "H"   :   0.10,
#  "He"  :   0.40,
#  "Li"  :   0.81,
#  "Be"  :   0.53,
#  "B"   :   0.92,
#  "C"   :   0.70,
#  "N"   :   0.55,
#  "O"   :   0.52,
#  "F"   :   0.47,
#  "Ne"  :   0.54,
#  "Na"  :   0.27,
#  "Mg"  :   0.73,
#  "Al"  :   0.84,
#  "Si"  :   0.10,
#  "P"   :   0.80,
#  "S"   :   0.80,
#  "Cl"  :   0.75,
#  "Ar"  :   0.88,
#  "K"   :   0.75,
#  "Ca"  :   0.31,
#  "Ga"  :   0.87,
#  "Ge"  :   0.11,
#  "As"  :   0.85,
#  "Se"  :   0.90,
#  "Br"  :   0.83,
#  "Kr"  :   0.02,
#  "Rb"  :   0.03,
#  "Sr"  :   0.49,
#  "In"  :   0.93,
#  "Sn"  :   0.17,
#  "Sb"  :   0.06,
#  "Te"  :   0.06,
#  "I"   :   0.98,
#  "Xe"  :   0.16,
#  "Cs"  :   0.43,
#  "Ba"  :   0.68,
#  "Tl"  :   0.96,
#  "Pb"  :   0.02,
#  "Bi"  :   0.07,
#  "Po"  :   0.97,
#  "At"  :   0.02,
#  "Rn"  :   0.20,
#  "Fr"  :   0.48,
#  "Ra"  :   0.83,
## Transition metals.  4s means the radius of the 4s shell was used,
## B means taken from Bondi DOI: 10.1246/bcsj.20100166
#  "Sc"  :   0.08,   # 4s
#  "Ti"  :   0.99,   # 4s
#  "V"   :   0.91,   # 4s
#  "Cr"  :   0.92,   # 4s
#  "Mn"  :   0.77,   # 4s
#  "Fe"  :   0.71,   # 4s
#  "Co"  :   0.65,   # 4s
#  "Ni"  :   0.63,   # B
#  "Cu"  :   0.40,   # B
#  "Zn"  :   0.39,   # B
#  "Y"   :   0.23,   # 5s
#  "Zr"  :   0.12,   # 5s
#  "Nb"  :   0.03,   # 5s
#  "Mo"  :   0.95,   # 5s
#  "Tc"  :   0.89,   # 5s
#  "Ru"  :   0.89,   # 5s
#  "Rh"  :   0.86,   # 5s
#  "Pd"  :   0.63,   # B
#  "Ag"  :   0.72,   # B
#  "Cd"  :   0.58    # B
# }

Radii = {
 "H"   :   1.10,
 "He"  :   1.40,
 "Li"  :   1.81,
 "Be"  :   1.53,
 "B"   :   1.92,
 "C"   :   1.70,
 "N"   :   1.55,
 "O"   :   1.52,
 "F"   :   1.47,
 "Ne"  :   1.54,
 "Na"  :   2.27,
 "Mg"  :   1.73,
 "Al"  :   1.84,
 "Si"  :   2.10,
 "P"   :   1.80,
 "S"   :   1.80,
 "Cl"  :   1.75,
 "Ar"  :   1.88,
 "K"   :   2.75,
 "Ca"  :   2.31,
 "Ga"  :   1.87,
 "Ge"  :   2.11,
 "As"  :   1.85,
 "Se"  :   1.90,
 "Br"  :   1.83,
 "Kr"  :   2.02,
 "Rb"  :   3.03,
 "Sr"  :   2.49,
 "In"  :   1.93,
 "Sn"  :   2.17,
 "Sb"  :   2.06,
 "Te"  :   2.06,
 "I"   :   1.98,
 "Xe"  :   2.16,
 "Cs"  :   3.43,
 "Ba"  :   2.68,
 "Tl"  :   1.96,
 "Pb"  :   2.02,
 "Bi"  :   2.07,
 "Po"  :   1.97,
 "At"  :   2.02,
 "Rn"  :   2.20,
 "Fr"  :   3.48,
 "Ra"  :   2.83,
# Transition metals.  4s means the radius of the 4s shell was used,
# B means taken from Bondi DOI: 10.1246/bcsj.20100166
 "Sc"  :   2.08,   # 4s
 "Ti"  :   1.99,   # 4s
 "V"   :   1.91,   # 4s
 "Cr"  :   1.92,   # 4s
 "Mn"  :   1.77,   # 4s
 "Fe"  :   1.71,   # 4s
 "Co"  :   1.65,   # 4s
 "Ni"  :   1.63,   # B
 "Cu"  :   1.40,   # B
 "Zn"  :   1.39,   # B
 "Y"   :   2.23,   # 5s
 "Zr"  :   2.12,   # 5s
 "Nb"  :   2.03,   # 5s
 "Mo"  :   1.95,   # 5s
 "Tc"  :   1.89,   # 5s
 "Ru"  :   1.89,   # 5s
 "Rh"  :   1.86,   # 5s
 "Pd"  :   1.63,   # B
 "Ag"  :   1.72,   # B
 "Cd"  :   1.58    # B
 }
