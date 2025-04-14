SHELL = /bin/sh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

.PHONY: default bin compile check clean realclean doDirs

DIRS = readPoPs checkIntids

default: compile

bin:
	echo ""
	echo "  -- PoPI/bin/Src --"
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
