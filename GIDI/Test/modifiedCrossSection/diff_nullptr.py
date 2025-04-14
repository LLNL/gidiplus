# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

import sys

sep = '# Reaction:'

def read(file):

    with open(file) as file:
        lines1 = ''.join(file.readlines())

    data = sep + sep.join(lines1.split(sep)[1:])
    return(data)

data1 = read(sys.argv[2])
data2 = read(sys.argv[3])

if data1 != data2: print('FAILURE: %s: difference in output.' % sys.argv[1])
