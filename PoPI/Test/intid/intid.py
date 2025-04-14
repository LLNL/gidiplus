# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

from PoPs import database as databaseModule
from PoPs import intId as intIdModule

pops = databaseModule.read('../../../TestData/PoPs/pops.xml')

with open('intid.out') as fIn:
    lines = fIn.readlines()

for line in lines:
    pid, intid, *dummy = line.split()
    particle = pops[pid]
    try:
        intid = particle.intid()
    except:
        print(pid)
        continue
    pid2 = intIdModule.idFromIntid(intid)
    print(pid, intid, pid2)
