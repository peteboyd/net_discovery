#!/usr/bin/env python
import pickle
import options
import sys
import os
import ConfigParser
from logging import info, debug, warning, error, critical
from options import clean
options = options.Options()
sys.path.append(options.faps_dir)
from faps import Structure
sys.path.append(options.genstruct_dir)
from SecondaryBuildingUnit import SBU
sys.path.append(options.max_clique_dir)
from data_storage import CSV, FunctionalGroups
from nets import Net

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

def pickler(options, dic, inchi=False):
    """write the dictionary to a pickle file"""
    pickle_name = clean(os.path.basename(options.input_file))
    pickle_name += "_inchi" if inchi else ""
    pickle_file = open(pickle_name+".pkl", "wb")
    pickle.dump(dic, pickle_file)
    pickle_file.close()

def main():
    mofs = CSV(options.csv_file)
    good_mofs = CSV(clean(options.input_file)+"_complete_nets.csv")
    bad_mofs = CSV(clean(options.input_file)+"_bad.csv")
    mofs.read()
    # read in sbus
    sbus = read_sbu_files(options)
    # read in functional groups
    fnls = FunctionalGroups(options)
    nets, inchikeys = {}, {}
    for mof_name in mofs.get('MOFname'):
        mof_name = clean(mof_name)
        mof = Structure(mof_name)
        try:
            mof.from_file(os.path.join(options.lookup,
                         mof_name), "cif", '')
        except IOError:
            mof.from_file(os.path.join(options.lookup,
                          mof_name+".out"), "cif", '')
        try:
            ff = fnls[mof_name]
        except KeyError:
            ff = (None, None)
        net = Net(options, mof)
        if options.mofs_from_groin:
            clq = net.from_groin_mof(sbus, ff)
        else:
            error("NO implementation yet for non-groin MOFs.")
            sys.exit()
        if net.evaluate_completeness():
            net.get_edges()
            net.get_nodes()
            # extra check if the unit cell is misrepresented.
            if net.prune_unit_cell():
                inchikeys.update(net.organic_data())
                net.pickle_prune()
                nets[net.name] = net
                good_mofs.add_data(MOFname=mof_name)
            else:
                bad_mofs.add_data(MOFname=mof_name)
        else:
            bad_mofs.add_data(MOFname=mof_name)

    pickler(options, nets)
    pickler(options, inchikeys, inchi=True)
    good_mofs.write()
    bad_mofs.write()

if __name__=="__main__":
    main()
