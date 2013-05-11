#!/usr/bin/env python
"""
Configuration options for genstruct.
"""
from optparse import OptionParser


class Options(object):
    """
    takes command line options and sets defaults for building structures.
    """
    def __init__(self):
        self.options = None 
        # files to read for database
        # option to read particular sbus from database
        # option to run different versions of the generation
        # option to ignore functional groups during generation
        # option to debug generation code (ie. periodically write to debug.xyz)
        # option to supress logging info
        self.commandline()
    def commandline(self):
        """ get options from command line, overwrite defaults """

        parser = OptionParser()
        parser.add_option("-c", "--choose", action="store_true",
                          dest="select_sbu",
                          help="select specific SBU's for building")
        parser.add_option("-v", "--verbose", action="store_true",
                          dest="verbose",
                          help="output debugging information")
        parser.add_option("-d", "--debug", action="store_true",
                          dest="debug",
                          help="print debugging files [debug.xyz] [history.xyz]")
        (options, args) = parser.parse_args()
        self.options = options
