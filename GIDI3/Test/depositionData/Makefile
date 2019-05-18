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

POPs_PATH = ../../../PoPsCpp
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
		-I$(POPs_INCLUDE) -L$(POPs_LIB) \
		-I$(SMR_INCLUDE) -L$(SMR_LIB) \
		-I$(NF_INCLUDE) -L$(NF_LIB) \
		-I$(PUGIXML_INCLUDE) -L$(PUGIXML_LIB)

LIBS = -lGIDI -lPoPs -lstatusMessageReporting -lnumericalfunctions -lpugixml

default: depositionEnergy depositionMomentum

depositionEnergy: depositionEnergy.cpp
	$(CXX) $(local_CXXFLAGS) -o depositionEnergy depositionEnergy.cpp $(LIBS)

depositionMomentum: depositionMomentum.cpp
	$(CXX) $(local_CXXFLAGS) -o depositionMomentum depositionMomentum.cpp $(LIBS)

check: depositionEnergy depositionMomentum
	./depositionEnergy O16 > o
	python ../Utilities/diff.py "depositionData/depositionEnergy O16" O16.energy.out o
	./depositionEnergy O16 n > o
	python ../Utilities/diff.py "depositionData/depositionEnergy O16 n" O16.n.energy.out o
	./depositionEnergy O16 n photon > o
	python ../Utilities/diff.py "depositionData/depositionEnergy O16 n photon" O16.n.photon.energy.out o
	./depositionEnergy Th227 > o
	python ../Utilities/diff.py "depositionData/depositionEnergy Th227" Th227.energy.out o
	./depositionEnergy Th227 n > o
	python ../Utilities/diff.py "depositionData/depositionEnergy Th227 n" Th227.n.energy.out o
	./depositionEnergy Th227 n photon > o
	python ../Utilities/diff.py "depositionData/depositionEnergy Th227 n photon" Th227.n.photon.energy.out o

	./depositionMomentum O16 > o
	python ../Utilities/diff.py "depositionData/depositionMomentum O16" O16.momentum.out o
	./depositionMomentum O16 n > o
	python ../Utilities/diff.py "depositionData/depositionMomentum O16 n" O16.n.momentum.out o
	./depositionMomentum O16 n photon > o
	python ../Utilities/diff.py "depositionData/depositionMomentum O16 n photon" O16.n.photon.momentum.out o
	./depositionMomentum Th227 > o
	python ../Utilities/diff.py "depositionData/depositionMomentum Th227" Th227.momentum.out o
	./depositionMomentum Th227 n > o
	python ../Utilities/diff.py "depositionData/depositionMomentum Th227 n" Th227.n.momentum.out o
	./depositionMomentum Th227 n photon > o
	python ../Utilities/diff.py "depositionData/depositionMomentum Th227 n photon" Th227.n.photon.momentum.out o


clean:
	rm -f depositionEnergy depositionMomentum o file1 file2

realclean: clean
	rm -rf *.dSYM
