SHELL = /bin/ksh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

.PHONY: default availableEnergy availableMomentum clean realclean

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

default: availableEnergy availableMomentum

availableEnergy: availableEnergy.cpp
	$(CXX) $(local_CXXFLAGS) -o availableEnergy availableEnergy.cpp $(LIBS)

availableMomentum: availableMomentum.cpp
	$(CXX) $(local_CXXFLAGS) -o availableMomentum availableMomentum.cpp $(LIBS)

check: availableEnergy availableMomentum
	./availableEnergy --tid O16 > o1
	python ../Utilities/diff.py "availableEnergy/availableEnergy O16" O16.energy.out o1
	./availableEnergy --tid Th227 > o2
	python ../Utilities/diff.py "availableEnergy/availableEnergy Th227" Th227.energy.out o2
	./availableMomentum --tid O16 > o3
	python ../Utilities/diff.py "availableEnergy/availableMomentum O16" O16.momentum.out o3
	./availableMomentum --tid Th227 > o4
	python ../Utilities/diff.py "availableEnergy/availableMomentum Th227" Th227.momentum.out o4

clean:
	rm -f availableEnergy availableMomentum o[0-9]* file1 file2

realclean: clean
	rm -rf *.dSYM
