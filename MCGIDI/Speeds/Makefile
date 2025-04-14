SHELL = /bin/sh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

DIR_BINS = crossSection crossSection_multiGroup sampleProducts sampleReactions
DIRS = Utilities $(DIR_BINS)

.PHONY: default check speeds clean realclean doDirs $(DIRS)

default:
	echo ""
	echo "  -- MCGIDI/Speeds --"
	$(MAKE) $(DIR_BINS)

check:

speeds:
	echo ""
	echo "  -- MCGIDI/Speeds --"
	$(MAKE) doDirs TARGET=speeds

Utilities:
	cd $@; $(MAKE)

$(DIR_BINS): Utilities
	cd $@; $(MAKE)

clean:
	$(MAKE) doDirs TARGET=clean

realclean:
	$(MAKE) doDirs TARGET=realclean

doDirs:
	for dir in $(DIRS); do cd $$dir; $(MAKE) $(TARGET); cd ../; done
