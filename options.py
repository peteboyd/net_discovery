#!/usr/bin/env python
import logging
from logging import debug, error, info
import os
import re
import sys
import copy
import textwrap
from ast import literal_eval
from optparse import OptionParser
import ConfigParser
from StringIO import StringIO

class Options(object):
    """Read in the options from the config file."""
    def __init__(self):
        # read in from the command line first
        self._command_options()
        self._init_logging()
        self.job = ConfigParser.SafeConfigParser()
        self._set_paths()
        self._load_defaults()
        self._load_job()
        self._set_attr()

    def _set_paths(self):
        if __name__ != '__main__':
            self.script_dir = os.path.dirname(__file__)
        else:
            self.script_dir = os.path.abspath(sys.path[0])
        self.job_dir = os.getcwd()

    def _load_defaults(self):
        """Load data from the default.ini in the code path."""
        default_path = os.path.join(self.script_dir, 'defaults.ini')
        try:
            filetemp = open(default_path, 'r')
            default = filetemp.read()
            filetemp.close()
            if not '[defaults]' in default.lower():
                default = '[defaults]\n' + default
            default = StringIO(default)
        except IOError:
            error("Error loading defaults.ini")
            default = StringIO('[defaults]\n')
        self.job.readfp(default)

    def _load_job(self):
        """Load data from the local job name."""
        if self.cmd_opts.input_file is not None:
            job_path = os.path.join(self.job_dir, self.cmd_opts.input_file)
            try:
                filetemp = open(job_path, 'r')
                job = filetemp.read()
                filetemp.close()
                if not '[job]' in job.lower():
                    job = '[job]\n' + job
                job = StringIO(job)
            except IOError:
                job = StringIO('[job]\n')
        else:
            job = StringIO('[job]\n')
        self.job.readfp(job)

    def _command_options(self):
        """Load data from the command line."""

        usage = "%prog [options]"
        parser = OptionParser(usage=usage)
        parser.add_option("-f", "--file", action="store", type="string",
                          dest="input_file", default="None",
                          help="Specify a job-specific input file.")
        parser.add_option("-C", "--combine", type="string",
                          dest="combine", action="callback",
                          callback=self.parse_commas,
                          help="comma (,) delimited list of csv files to " + \
                          "merge into a larger file.")
        parser.add_option("-e", "--extract", action="store",
                          dest='extract',
                          help="Extract info from the supplied file name.")
        parser.add_option("-Q", "--sqlfile", action="store",
                          dest="sql_file",
                          help="Specify the sql file containing all " + \
                                  "functional group information.")
        parser.add_option("-s", "--silent", action="store_true",
                          dest="silent",
                          help="Print nothing to the console.")
        parser.add_option("-q", "--quiet", action="store_true",
                          dest="quiet",
                          help="Print only warnings and errors.")
        parser.add_option("-v", "--verbose", action="store_true",
                          dest="verbose",
                          help="Print everything to the console.")
        (local_options, local_args) = parser.parse_args()
        #if len(sys.argv) == 1:
        #    parser.print_help()
        #    sys.exit(1)
        self.cmd_opts = local_options

    def _init_logging(self):
        """Initiate the logging"""
        if self.cmd_opts.silent:
            stdout_level = logging.CRITICAL
            file_level = logging.INFO
        elif self.cmd_opts.quiet:
            stdout_level = logging.ERROR
            file_level = logging.INFO
        elif self.cmd_opts.verbose:
            stdout_level = logging.DEBUG
            file_level = logging.DEBUG
        else:
            stdout_level = logging.INFO
            file_level = logging.INFO

        logging.basicConfig(level=file_level,
                            format='[%(asctime)s] %(levelname)s %(message)s',
                            datefmt='%Y%m%d %H:%M:%S',
                            filename="log.out",
                            filemode='a')

        logging.addLevelName(10, '--')
        logging.addLevelName(20, '>>')
        logging.addLevelName(30, '**')
        logging.addLevelName(40, '!!')
        logging.addLevelName(50, 'XX')

        console = ColouredConsoleHandler(sys.stdout)
        console.setLevel(stdout_level)
        formatter = logging.Formatter('%(levelname)s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    def parse_commas(self, option, opt, value, parser):
        setattr(parser.values, option.dest, value.split(','))

    def _set_attr(self):
        """Sets attributes to the base class. default options are over-written
        by job-specific options.

        """
        for key, value in self.job.items('defaults'):
            value = self.get_val('defaults', key)
            setattr(self, key, value)
        for key, value in self.job.items('job'):
            value = self.get_val('job', key)
            setattr(self, key, value)
        # special option to over-ride the input file, so that
        # extract can be easily issued from the command-line.
        if self.cmd_opts.extract:
            setattr(self, 'extract', True)
            setattr(self, 'csv_file', self.cmd_opts.extract)
        # option over-writes the sql file in the input file
        if self.cmd_opts.sql_file:
            setattr(self, 'sql_file', self.cmd_opts.sql_file)

    def get_val(self, section, key):
        """Returns the proper type based on the key used."""
        floats = ['tolerance']
        booleans = ['mofs_from_groin']
        integers = ['report_frequency']
        lists = ['sbu_files', 'ignore_list']
        tuples = ['supercell']
        # known booleans
        if key in booleans: 
            try:
                val = self.job.getboolean(section, key)
            except ValueError:
                val = False
        # known integers
        elif key in integers: 
            try: 
                val = self.job.getint(section, key)
            except ValueError:
                val = 0
        # known floats
        elif key in floats: 
            try:
                val = self.job.getfloat(section, key)
            except ValueError:
                val = 0.
        # known lists
        elif key in lists:
            p = re.compile('[,;\s]+')
            val = p.split(self.job.get(section, key))
            val = [i for i in val if i]

        # known tuples
        elif key in tuples:
            val = literal_eval(self.job.get(section, key)) 

        else:
            val = self.job.get(section, key)
        return val

class ColouredConsoleHandler(logging.StreamHandler):
    """Makes colourised and wrapped output for the console."""
    def emit(self, record):
        """Colourise and emit a record."""
        # Need to make a actual copy of the record
        # to prevent altering the message for other loggers
        myrecord = copy.copy(record)
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
