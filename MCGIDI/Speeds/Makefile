SHELL = /bin/ksh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

.PHONY: default check speeds clean realclean doDirs

DIRS = Utilities crossSection crossSection_multiGroup sampleProducts sampleReactions

default:
	echo ""
	echo "  -- MCGIDI/Speeds --"
	$(MAKE) doDirs TARGET=default

check:

speeds:
	echo ""
	echo "  -- MCGIDI/Speeds --"
	$(MAKE) doDirs TARGET=speeds

clean:
	$(MAKE) doDirs TARGET=clean

realclean:
	$(MAKE) doDirs TARGET=realclean

doDirs:
	for dir in $(DIRS); do cd $$dir; $(MAKE) $(TARGET); cd ../; done
