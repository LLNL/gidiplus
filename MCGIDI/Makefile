SHELL = /bin/ksh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

DIRS = Src include lib bin Doc

POPs_PATH = $(abspath ../PoPI)
SMR_PATH = $(abspath ../statusMessageReporting)
NF_PATH = $(abspath ../numericalFunctions)

.PHONY: default include lib bin clean realclean

default: include lib
	cd Test; $(MAKE) default
	$(MAKE) bin

include:
	cd include; $(MAKE)

lib:
	cd lib; $(MAKE) CXX=$(CXX) CXXFLAGS='$(CXXFLAGS)' POPs_PATH='$(POPs_PATH)' SMR_PATH='$(SMR_PATH)' NF_PATH='$(NF_PATH)'

bin:
	cd bin; $(MAKE) CXX=$(CXX) CXXFLAGS='$(CXXFLAGS)' POPs_PATH='$(POPs_PATH)' SMR_PATH='$(SMR_PATH)' NF_PATH='$(NF_PATH)' bin

check:
	cd Test; $(MAKE) CXX=$(CXX) CXXFLAGS='$(CXXFLAGS)' POPs_PATH='$(POPs_PATH)' SMR_PATH='$(SMR_PATH)' NF_PATH='$(NF_PATH)' check

speeds:
	cd Speeds; $(MAKE) CXX=$(CXX) CXXFLAGS='$(CXXFLAGS)' POPs_PATH='$(POPs_PATH)' SMR_PATH='$(SMR_PATH)' NF_PATH='$(NF_PATH)' speeds

clean:
	$(MAKE) doDIRS _DIRS="$(DIRS)" _TARGET=clean
	cd Test; $(MAKE) clean
	cd Speeds; $(MAKE) clean

realclean:
	$(MAKE) doDIRS _DIRS="$(DIRS)" _TARGET=realclean
	cd Test; $(MAKE) realclean
	cd Speeds; $(MAKE) realclean

docs:
	doxygen Doc/doxygen/Doxyfile

doDIRS:
	SAVED_PWD=`pwd`; \
	for DIR in $(_DIRS); do cd $$DIR; $(MAKE) $(_TARGET); cd $$SAVED_PWD; done
