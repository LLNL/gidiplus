SHELL = /bin/ksh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

.PHONY: default map nesting libraries clean realclean

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
        -I../Utilities ../Utilities/GIDI_testUtilities.o \
		-I$(POPs_INCLUDE) -L$(POPs_LIB) \
		-I$(SMR_INCLUDE) -L$(SMR_LIB) \
		-I$(NF_INCLUDE) -L$(NF_LIB) \
        -I$(PUGIXML_INCLUDE) -L$(PUGIXML_LIB)

default: map nesting libraries

map: map.cpp
	$(CXX) $(local_CXXFLAGS) -o map map.cpp -lGIDI -lPoPI -lstatusMessageReporting -lnumericalfunctions -lpugixml

nesting: nesting.cpp
	$(CXX) $(local_CXXFLAGS) -o nesting nesting.cpp -lGIDI -lPoPI -lstatusMessageReporting -lnumericalfunctions -lpugixml

libraries: libraries.cpp
	$(CXX) $(local_CXXFLAGS) -o libraries libraries.cpp -lGIDI -lPoPI -lstatusMessageReporting -lnumericalfunctions -lpugixml

check: map nesting libraries
	./map > o
	../Utilities/diff.com map/map map.out o
	./nesting > o
	../Utilities/diff.com map/nesting nesting.out o
	./libraries > o
	../Utilities/diff.com map/libraries libraries.out o

clean:
	rm -f map nesting libraries o file1 file2

realclean: clean
	rm -rf *.dSYM
