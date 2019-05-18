SHELL = /bin/ksh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

DIRS = Src

.PHONY: default bin compile check clean realclean

default: compile

bin:
	$(MAKE) doDirs TARGET=bin CXX=$(CXX) CXXFLAGS="$(CXXFLAGS)"

compile:
	$(MAKE) doDirs TARGET=default CXX=$(CXX) CXXFLAGS="$(CXXFLAGS)"

check:

clean:
	$(MAKE) doDirs TARGET=clean CXX=$(CXX) CXXFLAGS="$(CXXFLAGS)"
	rm -f readAllProtaresInMapFile directory

realclean:
	$(MAKE) doDirs TARGET=realclean CXX=$(CXX) CXXFLAGS="$(CXXFLAGS)"
	rm -f readAllProtaresInMapFile directory toXML

doDirs:
	for dir in $(DIRS); do cd $$dir; $(MAKE) $(TARGET) CXX=$(CXX) CXXFLAGS="$(CXXFLAGS)"; cd ../; done
