SHELL = /bin/sh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

.PHONY: default compile check clean realclean doDirs

DIRS = alias constructing help missingValue positionalMultipleWithVariousRequired sameName statusMessageReporting splitString

default:

compile:
	$(MAKE) doDirs TARGET=default

check:
	echo ""
	echo "  -- Checking LUPI --"
	$(MAKE) doDirs TARGET=check

clean:
	$(MAKE) doDirs TARGET=clean

realclean:
	$(MAKE) doDirs TARGET=realclean

doDirs:
	for dir in $(DIRS); do cd $$dir; $(MAKE) $(TARGET); cd ../; done
