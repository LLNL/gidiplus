SHELL = /bin/sh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

DIRS = readAllProtaresInMapFile angleBiasing broomstick

.PHONY: default bin compile check clean realclean doDirs $(DIRS)

default: bin

bin:
	echo ""
	echo "  -- MCGIDI/bin/Src --"
	$(MAKE) $(DIRS)

compile:
	$(MAKE) doDirs TARGET=bin

check:

$(DIRS):
	cd $@; $(MAKE) bin

clean:
	$(MAKE) doDirs TARGET=clean

realclean:
	$(MAKE) doDirs TARGET=realclean

doDirs:
	for dir in $(DIRS); do cd $$dir; $(MAKE) $(TARGET); cd ../; done
