SHELL = /bin/bash

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

.PHONY: default include clean realclean

DIRS = statusMessageReporting GIDI3 PoPsCpp numericalFunctions pugixml

default: include

include:
	cp ../gidiplus_version.h .
	for DIR in $(DIRS); do cd ../$$DIR/; $(MAKE) include; done
	for DIR in $(DIRS); do cp ../$$DIR/include/*.h* .; done

clean:
	rm -f *.h *.hpp

realclean: clean
