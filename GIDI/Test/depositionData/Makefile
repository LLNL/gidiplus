SHELL = /bin/ksh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

.PHONY: default depositionEnergy depositionMomentum clean realclean

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

default: depositionEnergy depositionMomentum

depositionEnergy: depositionEnergy.cpp
	$(CXX) $(local_CXXFLAGS) -o depositionEnergy depositionEnergy.cpp $(LIBS)

depositionMomentum: depositionMomentum.cpp
	$(CXX) $(local_CXXFLAGS) -o depositionMomentum depositionMomentum.cpp $(LIBS)

check: depositionEnergy depositionMomentum
	./depositionEnergy --tid O16 > o1
	python ../Utilities/diff.py "depositionData/depositionEnergy O16" O16.energy.out o1
	./depositionEnergy --tid O16 n > o2
	python ../Utilities/diff.py "depositionData/depositionEnergy O16 n" O16.n.energy.out o2
	./depositionEnergy --tid O16 n photon > o3
	python ../Utilities/diff.py "depositionData/depositionEnergy O16 n photon" O16.n.photon.energy.out o3
	./depositionEnergy --tid Th227 > o4
	python ../Utilities/diff.py "depositionData/depositionEnergy Th227" Th227.energy.out o4
	./depositionEnergy --tid Th227 n > o5
	python ../Utilities/diff.py "depositionData/depositionEnergy Th227 n" Th227.n.energy.out o5
	./depositionEnergy --tid Th227 n photon > o6
	python ../Utilities/diff.py "depositionData/depositionEnergy Th227 n photon" Th227.n.photon.energy.out o6

	./depositionMomentum --tid O16 > o7
	python ../Utilities/diff.py "depositionData/depositionMomentum O16" O16.momentum.out o7
	./depositionMomentum --tid O16 n > o8
	python ../Utilities/diff.py "depositionData/depositionMomentum O16 n" O16.n.momentum.out o8
	./depositionMomentum --tid O16 n photon > o9
	python ../Utilities/diff.py "depositionData/depositionMomentum O16 n photon" O16.n.photon.momentum.out o9
	./depositionMomentum --tid Th227 > o10
	python ../Utilities/diff.py "depositionData/depositionMomentum Th227" Th227.momentum.out o10
	./depositionMomentum --tid Th227 n > o11
	python ../Utilities/diff.py "depositionData/depositionMomentum Th227 n" Th227.n.momentum.out o11
	./depositionMomentum --tid Th227 n photon > o12
	python ../Utilities/diff.py "depositionData/depositionMomentum Th227 n photon" Th227.n.photon.momentum.out o12

clean:
	rm -f depositionEnergy depositionMomentum o[0-9]* file1 file2

realclean: clean
	rm -rf *.dSYM
