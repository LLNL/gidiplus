SHELL = /bin/bash

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

.PHONY: default headers clean realclean

DIRS = statusMessageReporting GIDI3 MCGIDI3 RCGIDI PoPsCpp numericalFunctions pugixml

default: libs

libs:
	rm -f *.a *.o
	for DIR in $(DIRS); do cp ../$$DIR/lib/*.a* .; done
	archives=`/bin/ls *.a`; for archive in $$archives; do $(AR) -x $$archive; done
	$(AR) -cr libgidiplus.a *.o
	rm -rf *.o

clean:
	rm -f *.a *.o

realclean: clean
