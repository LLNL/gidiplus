SHELL = /bin/ksh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

.PHONY: default bin compile check clean realclean doDirs

DIRS = readAllProtaresInMapFile angleBiasing

default: bin

bin:
	echo ""
	echo "  -- MCGIDI/bin/Src --"
	$(MAKE) doDirs TARGET=bin

compile:
	$(MAKE) doDirs TARGET=bin

check:

clean:
	$(MAKE) doDirs TARGET=clean

realclean:
	$(MAKE) doDirs TARGET=realclean

doDirs:
	for dir in $(DIRS); do cd $$dir; $(MAKE) $(TARGET); cd ../; done
