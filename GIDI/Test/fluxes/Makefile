SHELL = /bin/ksh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

.PHONY: default fluxes bdfls3dFlux clean realclean

PUGIXML_PATH = ../../../pugixml
PUGIXML_INCLUDE = $(PUGIXML_PATH)/include
PUGIXML_LIB = $(PUGIXML_PATH)/lib

POPs_PATH = ../../../PoPI
POPs_INCLUDE = $(POPs_PATH)/include
POPs_LIB = $(POPs_PATH)/lib

SMR_PATH = ../../../statusMessageReporting
SMR_INCLUDE = $(SMR_PATH)/include
SMR_LIB = $(SMR_PATH)/lib

NF_PATH = ../../../numericalFunctions
NF_INCLUDE = $(NF_PATH)/include
NF_LIB = $(NF_PATH)/lib

local_CXXFLAGS = $(CXXFLAGS) -I../../include -L../../lib \
		-I$(POPs_INCLUDE) -L$(POPs_LIB) \
		-I../Utilities -L../Utilities \
		-I$(SMR_INCLUDE) -L$(SMR_LIB) \
		-I$(NF_INCLUDE) -L$(NF_LIB) \
        -I$(PUGIXML_INCLUDE) -L$(PUGIXML_LIB)

LIBS = -lGIDI_testUtilities -lGIDI -lPoPI -lstatusMessageReporting -lnumericalfunctions -lpugixml

default: fluxes bdfls3dFlux

fluxes: fluxes.cpp
	$(CXX) $(local_CXXFLAGS) -o fluxes fluxes.cpp $(LIBS)

bdfls3dFlux: bdfls3dFlux.cpp
	$(CXX) $(local_CXXFLAGS) -o bdfls3dFlux bdfls3dFlux.cpp $(LIBS)

check: fluxes bdfls3dFlux
	./fluxes > o1
	diff fluxes.out o1 > /dev/null; if [ $$? -ne 0 ]; then echo "FAILURE: fluxes/fluxes.cpp: difference in output"; fi
	./bdfls3dFlux > o2
	diff bdfls3dFlux.out o2 > /dev/null; if [ $$? -ne 0 ]; then echo "FAILURE: bdfls3dFlux/bdfls3dFlux.cpp: difference in output"; fi

clean:
	rm -f fluxes bdfls3dFlux o[0-9]* file1 file2

realclean: clean
	rm -rf *.dSYM
