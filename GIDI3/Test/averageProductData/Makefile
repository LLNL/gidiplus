SHELL = /bin/ksh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

.PHONY: default averageEnergy averageMomentum clean realclean

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
		-I$(POPs_INCLUDE) -L$(POPs_LIB) \
		-I$(SMR_INCLUDE) -L$(SMR_LIB) \
		-I$(NF_INCLUDE) -L$(NF_LIB) \
        -I$(PUGIXML_INCLUDE) -L$(PUGIXML_LIB)

LIBS = -lGIDI -lPoPs -lstatusMessageReporting -lnumericalfunctions -lpugixml

default: averageEnergy averageMomentum

averageEnergy: averageEnergy.cpp
	$(CXX) $(local_CXXFLAGS) -o averageEnergy averageEnergy.cpp $(LIBS)

averageMomentum: averageMomentum.cpp
	$(CXX) $(local_CXXFLAGS) -o averageMomentum averageMomentum.cpp $(LIBS)

check: averageEnergy averageMomentum
	./averageEnergy O16 > o
	python ../Utilities/diff.py "averageProductData/averageEnergy O16" O16.energy.out o
	./averageEnergy Th227 > o
	python ../Utilities/diff.py "averageProductData/averageEnergy Th227" Th227.energy.out o
	./averageMomentum O16 > o
	python ../Utilities/diff.py "averageProductData/averageMomentum O16" O16.momentum.out o
	./averageMomentum Th227 > o
	python ../Utilities/diff.py "averageProductData/averageMomentum Th227" Th227.momentum.out o

clean:
	rm -f averageEnergy averageMomentum o file1 file2

realclean: clean
	rm -rf *.dSYM
