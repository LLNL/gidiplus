SHELL = /bin/ksh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

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

local_CXXFLAGS = $(CXXFLAGS) \
        -I../../include -I$(POPs_INCLUDE) -I$(SMR_INCLUDE) -I$(NF_INCLUDE) -I$(PUGIXML_INCLUDE)

.PHONY: default check clean realclean

default: libGIDI_testUtilities.a

GIDI_testUtilities.o: GIDI_testUtilities.cpp
	$(CXX) $(local_CXXFLAGS) -c -o GIDI_testUtilities.o GIDI_testUtilities.cpp

libGIDI_testUtilities.a: GIDI_testUtilities.o
	$(AR) -cr libGIDI_testUtilities.a *.o

check: libGIDI_testUtilities.a

clean:
	rm -f GIDI_testUtilities.o file1 file2

realclean: clean
	rm -rf *.dSYM
