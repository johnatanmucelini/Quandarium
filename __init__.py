"""
Quandarium
==========

Provides
  1. Tools to deal with atomistic material simulation packages output.
  2. Tools to analyse and mine data from those simulations.
  3. A set of predefine tools to plot data to human interpretation.

This package present several tools to processe KDD for QC data and
calculations.
In particular, this package focus on data extraction, feature
enginering, DM processes, and data plot."""

import pkgutil
import logging

logging.basicConfig(filename='/home/johnatan/quandarium_module.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(module)s - %(funcName)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.info('The logging level is info')
logging.info(f'{"#"*20} QUANDARIUM WERE INITILIZED {"#"*20}')


# Importing all the submodules py files
__path__ = pkgutil.extend_path(__path__, __name__)
for importer, modname, ispkg in pkgutil.walk_packages(path=__path__,
                                                      prefix=__name__+'.'):
    __import__(modname)
