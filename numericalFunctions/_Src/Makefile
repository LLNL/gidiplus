# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

SHELL = /bin/sh

.PHONY: default collectSources clean realclean

SOURCES = ptwC/Src ptwX/Src ptwXY/Src nf_Legendre/Src nf_integration/Src nf_specialFunctions/Src

default:

collectSources: clean
	for source in $(SOURCES); do cp ../$$source/*.[ch] .; done

clean:
	rm -f *.c *.h

realclean: clean
