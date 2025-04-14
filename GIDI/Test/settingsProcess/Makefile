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
	./settingsProcess --tid O16 > Outputs/O16.out
	$(PYTHON) ../Utilities/diff.py settingsProcess/settingsProcess Benchmarks/O16.out Outputs/O16.out
	./settingsProcess --tid Th227 > Outputs/Th227.out
	$(PYTHON) ../Utilities/diff.py settingsProcess/settingsProcess Benchmarks/Th227.out Outputs/Th227.out
	./productMatrixCollapse --tid O16 > Outputs/productMatrixCollapse.O16.out
	$(PYTHON) ../Utilities/diff.py settingsProcess/productMatrixCollapse Benchmarks/productMatrixCollapse.O16.out Outputs/productMatrixCollapse.O16.out
	./productMatrixCollapse --tid Th227 > Outputs/productMatrixCollapse.Th227.out
	$(PYTHON) ../Utilities/diff.py settingsProcess/productMatrixCollapse Benchmarks/productMatrixCollapse.Th227.out Outputs/productMatrixCollapse.Th227.out
	./productMatrixCollapse_photoAtomic > Outputs/productMatrixCollapse_photoAtomic.O.out
	$(PYTHON) ../Utilities/diff.py settingsProcess/productMatrixCollapse_photoAtomic Benchmarks/productMatrixCollapse_photoAtomic.O.out Outputs/productMatrixCollapse_photoAtomic.O.out
