SHELL = /bin/ksh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

DIRS = Utilities matrix VectorAndMatrixReverse map parseXML groupBoundaries inverseSpeed productIDs crossSections productMatrix \
	transportCorrection availableEnergy multiplicity Q averageProductData depositionData \
	settingsProcess multiGroup temperatures splitString ancestry readAll photoAtomicScattering photoScattering TNSL gain

.PHONY: default check clean realclean

default:
	cd Utilities; $(MAKE) CXX=$(CXX) CXXFLAGS="$(CXXFLAGS)"

compile:
	$(MAKE) doDirs TARGET=default CXX=$(CXX) CXXFLAGS="$(CXXFLAGS)"

check:
	gunzip pops.xml.gz
	$(MAKE) doDirs TARGET=check CXX=$(CXX) CXXFLAGS="$(CXXFLAGS)"

clean:
	$(MAKE) doDirs TARGET=clean CXX=$(CXX) CXXFLAGS="$(CXXFLAGS)"

realclean:
	$(MAKE) doDirs TARGET=realclean CXX=$(CXX) CXXFLAGS="$(CXXFLAGS)"

doDirs:
	for dir in $(DIRS); do cd $$dir; $(MAKE) $(TARGET) CXX=$(CXX) CXXFLAGS="$(CXXFLAGS)"; cd ../; done
