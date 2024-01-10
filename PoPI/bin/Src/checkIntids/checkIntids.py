#! /usr/bin/env python3

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

import argparse
import pathlib

from PoPs import database as databaseModule
from PoPs import alias as aliasModule

description = '''Loops over each particle in the specified PoPs files, and prints each particles id and intid.'''

parser = argparse.ArgumentParser(description=description)

parser.add_argument('pops', type=pathlib.Path, nargs='+',           help='The list of pops files whom intid data are printed.')
args = parser.parse_args()

pops = None
for file in args.pops:
    if pops is None:
        pops = databaseModule.read(file)
    else:
        pops.addFile(file)

if pops is not None:
    for particle in pops:
        try:
            intid = particle.intid()
        except:
            intid = -1
        print(particle.id, intid, end='')

        particleFinal = pops.final(particle.id)
        if particle.id != particleFinal.id:
            print('', particleFinal.id, particleFinal.intid(), end='')
        print()
