#! /usr/bin/env python3

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

with open('test.ris') as fIn:
    lines = fIn.readlines()

for line in lines:
    print('<%s>' % line[:-1])
    for entry in [element.strip() for element in line.split(': 2')]:
        print('    <%s>' % entry)
