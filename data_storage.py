import os
from function_switch import FunctionalGroupLibrary, FunctionalGroup
from sub_graphs import SubGraph

class FunctionalGroups(object):
    
    def __init__(self, options, moflist=None):
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
            self._get_functional_group_lookup(moflist)

    def _get_functional_subgraphs(self):
        fnl_lib = FunctionalGroupLibrary()
        for name, obj in fnl_lib.items():
            if name != "H":
                sub = SubGraph(self.options, name)
                sub.from_fnl(obj)
                self._groups[name] = sub

    def _get_functional_group_lookup(self, moflist):
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
            valid = False
            if moflist:
                if mof in moflist:
                    valid = True
            else:
                valid = True
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
            if valid:
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

class CSV(object):
    """
    Reads in a .csv file for data parsing.

    """
    def __init__(self, filename):
        self.filename = filename
        self._data = {}
        self.headings = []

    def add_data(self, **kwargs):
        for key, value in kwargs.items():
            try:
                self.headings.index(key)
                self._data[key].append(value)
            except ValueError:
                self.headings.append(key)
                self._data[key] = [value]

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

    def read(self):
        """The CSV dictionary will store data to heading keys"""
        if not os.path.isfile(self.filename):
            error("Could not find the file: %s"%self.filename)
            sys.exit(1)
        filestream = open(self.filename, "r")
        head_line = filestream.readline().strip()
        self.headings = [i for i in head_line.lstrip("#").split(",") if i]
        for line in filestream:
            line = line.strip()
            if not line.startswith("#"):
                line = [i for i in line.split(",") if i]
                for ind, entry in enumerate(line):
                    # convert to float if possible
                    try:
                        entry = float(entry)
                    except ValueError:
                        #probably a string
                        pass
                    self._data.setdefault(self.headings[ind], []).append(entry)
        filestream.close()
    
    def write(self):
        string = ",".join(self.headings) + "\n"
        for line in zip(*[self._data[i] for i in self.headings]):
            string += ",".join(line) + "\n"
        file = open(self.filename, "w")
        file.writelines(string)
        file.close()

    def get(self, column):
        assert column in self._data.keys()
        return self._data[column]

