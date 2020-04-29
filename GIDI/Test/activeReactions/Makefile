SHELL = /bin/ksh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

.PHONY: default activeReactions clean realclean

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
		-I../../include -L../../lib \
		-I../Utilities -L../Utilities \
		-I$(POPs_INCLUDE) -L$(POPs_LIB) \
		-I$(SMR_INCLUDE) -L$(SMR_LIB) \
		-I$(NF_INCLUDE) -L$(NF_LIB) \
		-I$(PUGIXML_INCLUDE) -L$(PUGIXML_LIB)

LIBS = -lGIDI_testUtilities -lGIDI -lPoPI -lstatusMessageReporting -lnumericalfunctions -lpugixml

default: activeReactions

activeReactions: activeReactions.cpp
	$(CXX) $(local_CXXFLAGS) -o activeReactions activeReactions.cpp $(LIBS)

check: activeReactions
	./activeReactions --tracking photon --tracking n --tracking photon --tracking photon               > a 
	./activeReactions --tracking photon --tracking n --tracking photon --tracking photon -r 0          > b
	./activeReactions --tracking photon --tracking n --tracking photon --tracking photon -r 0 --invert > c
	./activeReactions.py activeReactions/activeReactions a c b
	./activeReactions  --tracking photon --tracking n --tracking photon --tracking photon -C 11 -C 40 -C 41 -C 42 -C 45 -C 48 -C 37 -C 46          > b
	./activeReactions  --tracking photon --tracking n --tracking photon --tracking photon -C 11 -C 40 -C 41 -C 42 -C 45 -C 48 -C 37 -C 46 --invert > c

clean:
	rm -f activeReactions

realclean: clean
	rm -rf *.dSYM a c b
