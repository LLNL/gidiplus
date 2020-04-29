SHELL = /bin/bash

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

DIRS = Utilities crossSection crossSection_multiGroup sampleProducts sampleReactions

.PHONY: default check speeds clean realclean

default:
	$(MAKE) doDirs TARGET=default CXX=$(CXX) CXXFLAGS="$(CXXFLAGS)"

check:

speeds:
	$(MAKE) doDirs TARGET=speeds CXX=$(CXX) CXXFLAGS="$(CXXFLAGS)"

clean:
	$(MAKE) doDirs TARGET=clean CXX=$(CXX) CXXFLAGS="$(CXXFLAGS)"

realclean:
	$(MAKE) doDirs TARGET=realclean CXX=$(CXX) CXXFLAGS="$(CXXFLAGS)"

doDirs:
	for dir in $(DIRS); do cd $$dir; $(MAKE) $(TARGET) CXX=$(CXX) CXXFLAGS="$(CXXFLAGS)"; cd ../; done
