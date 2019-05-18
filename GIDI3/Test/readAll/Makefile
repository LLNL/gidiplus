SHELL = /bin/ksh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

.PHONY: default readAll clean realclean

PUGIXML_PATH = ../../../pugixml
PUGIXML_INCLUDE = $(PUGIXML_PATH)/include
PUGIXML_LIB = $(PUGIXML_PATH)/lib

POPs_PATH = ../../../PoPsCpp
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

LIBS = -lGIDI -lPoPs -lstatusMessageReporting -lnumericalfunctions -lpugixml

default: readAll

readAll: readAll.cpp
	$(CXX) $(local_CXXFLAGS) -o readAll readAll.cpp $(LIBS)

check: readAll
	./readAll > o
	../Utilities/diff.com readAll/readAll all.out o
	./readAll 1 > o
	../Utilities/diff.com readAll/readAll all_maps.out o

clean:
	rm -f readAll o

realclean: clean
	rm -rf *.dSYM
