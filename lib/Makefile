SHELL = /bin/ksh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

.PHONY: default headers docs clean realclean

DIRS = GIDI MCGIDI PoPI HAPI numericalFunctions pugixml

default: libs

libs:
	rm -f *.a *.o
	for DIR in $(DIRS); do cp ../$$DIR/lib/*.a* .; done
	archives=`/bin/ls *.a`; for archive in $$archives; do $(AR) -x $$archive; done
	$(AR) -cr libgidiplus.a *.o
	rm -rf *.o

docs:

check:

clean:
	rm -f *.a *.o

realclean: clean
	rm -rf __.SYMDEF*
