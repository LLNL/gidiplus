SHELL = /bin/ksh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

.PHONY: default check clean realclean

DIRS = Test1 Test2 Test3 Test4 Test5 specialParticleID aliasReferences symbols baseAntiQualifierFromID intid

default:
	$(MAKE) doDirs TARGET=default

check:
	echo ""
	echo "  -- Checking PoPI --"
	$(MAKE) doDirs TARGET=check

clean:
	$(MAKE) doDirs TARGET=clean

realclean:
	$(MAKE) doDirs TARGET=realclean

doDirs:
	for dir in $(DIRS); do cd $$dir; $(MAKE) $(TARGET); cd ../; done
