SHELL = /bin/ksh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

.PHONY: default multiplicity clean realclean

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
		-I../Utilities -L../Utilities \
		-I$(POPs_INCLUDE) -L$(POPs_LIB) \
		-I$(SMR_INCLUDE) -L$(SMR_LIB) \
		-I$(NF_INCLUDE) -L$(NF_LIB) \
        -I$(PUGIXML_INCLUDE) -L$(PUGIXML_LIB)

LIBS = -lGIDI_testUtilities -lGIDI -lPoPI -lstatusMessageReporting -lnumericalfunctions -lpugixml

default: multiplicity

multiplicity: multiplicity.cpp
	$(CXX) $(local_CXXFLAGS) -o multiplicity multiplicity.cpp $(LIBS)

check: multiplicity
	./multiplicity --tid O16 > o1
	../Utilities/diff.com multiplicity/multiplicity O16.out o1
	./multiplicity --tid Th227 > o2
	../Utilities/diff.com multiplicity/multiplicity Th227.out o2

clean:
	rm -f multiplicity o[0-9]* file1 file2

realclean: clean
	rm -rf *.dSYM
