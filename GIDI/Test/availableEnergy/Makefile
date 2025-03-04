SHELL = /bin/sh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

GIDI_PLUS_PATH ?= $(abspath ../../..)
CppSource = $(sort $(wildcard *.cpp))

include $(GIDI_PLUS_PATH)/Makefile.paths
include ../Makefile.check

check: $(Executables)
	if [ ! -e Outputs ]; then mkdir Outputs; fi
	./availableEnergy --tid O16 > Outputs/O16.energy.out
	$(PYTHON) ../Utilities/diff.py "availableEnergy/availableEnergy O16" Benchmarks/O16.energy.out Outputs/O16.energy.out
	./availableEnergy --tid Th227 > Outputs/Th227.energy.out
	$(PYTHON) ../Utilities/diff.py "availableEnergy/availableEnergy Th227" Benchmarks/Th227.energy.out Outputs/Th227.energy.out
	./availableMomentum --tid O16 > Outputs/O16.momentum.out
	$(PYTHON) ../Utilities/diff.py "availableEnergy/availableMomentum O16" Benchmarks/O16.momentum.out Outputs/O16.momentum.out
	./availableMomentum --tid Th227 > Outputs/Th227.momentum.out
	$(PYTHON) ../Utilities/diff.py "availableEnergy/availableMomentum Th227" Benchmarks/Th227.momentum.out Outputs/Th227.momentum.out
