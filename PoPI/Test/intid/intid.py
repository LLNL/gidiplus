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
