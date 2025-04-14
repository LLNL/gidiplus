SHELL = /bin/sh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

.PHONY: default bin all include docs check clean realclean

DIRS = pugixml numericalFunctions LUPI Python HAPI GUPI CADI RISI PoPI GIDI MCGIDI

default: include

bin:

all: default

include:
	cp ../gidiplus_version.h .
	for DIR in $(DIRS); do cd ../$$DIR/; $(MAKE) include; done
	for DIR in $(DIRS); do cp ../$$DIR/include/*.h* .; done

docs:

check:

clean:
	rm -f *.h *.hpp

realclean: clean
