SHELL = /bin/bash

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

DIRS = Utilities domainHash crossSection sampleReactions sampleProducts \
		crossSection_multiGroup sampleReactions_multiGroup sampleProducts_multiGroup deposition_multiGroup \
		deposition_continuousEnergy productIndices samplePhotoAtomic sampleTerrellPromptNeutronDistribution \
		excludeReactions TNSL

.PHONY: default check clean realclean

default:
	cd Utilities; $(MAKE) CXX=$(CXX) CXXFLAGS="$(CXXFLAGS)"

check:
	$(MAKE) doDirs TARGET=check CXX=$(CXX) CXXFLAGS="$(CXXFLAGS)"

clean:
	$(MAKE) doDirs TARGET=clean CXX=$(CXX) CXXFLAGS="$(CXXFLAGS)"

realclean:
	$(MAKE) doDirs TARGET=realclean CXX=$(CXX) CXXFLAGS="$(CXXFLAGS)"

doDirs:
	for dir in $(DIRS); do cd $$dir; $(MAKE) $(TARGET) CXX=$(CXX) CXXFLAGS="$(CXXFLAGS)"; cd ../; done
