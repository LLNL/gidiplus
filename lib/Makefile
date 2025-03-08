SHELL = /bin/sh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

.PHONY: default bin all libs docs check clean realclean

DIRS = pugixml numericalFunctions LUPI Python HAPI GUPI CADI RISI PoPI GIDI MCGIDI

default: libs

bin:

all: default

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
