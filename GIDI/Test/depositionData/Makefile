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
	./depositionEnergy --tid O16 > Outputs/O16.energy.out
	$(PYTHON) ../Utilities/diff.py "depositionData/depositionEnergy O16" Benchmarks/O16.energy.out Outputs/O16.energy.out
	./depositionEnergy --tid O16 n > Outputs/O16.n.energy.out
	$(PYTHON) ../Utilities/diff.py "depositionData/depositionEnergy O16 n" Benchmarks/O16.n.energy.out Outputs/O16.n.energy.out
	./depositionEnergy --tid O16 n photon > Outputs/O16.n.photon.energy.out 
	$(PYTHON) ../Utilities/diff.py "depositionData/depositionEnergy O16 n photon" Benchmarks/O16.n.photon.energy.out Outputs/O16.n.photon.energy.out
	./depositionEnergy --tid Th227 > Outputs/Th227.energy.out
	$(PYTHON) ../Utilities/diff.py "depositionData/depositionEnergy Th227" Benchmarks/Th227.energy.out Outputs/Th227.energy.out
	./depositionEnergy --tid Th227 n > Outputs/Th227.n.energy.out
	$(PYTHON) ../Utilities/diff.py "depositionData/depositionEnergy Th227 n" Benchmarks/Th227.n.energy.out Outputs/Th227.n.energy.out
	./depositionEnergy --tid Th227 n photon > Outputs/Th227.n.photon.energy.out
	$(PYTHON) ../Utilities/diff.py "depositionData/depositionEnergy Th227 n photon" Benchmarks/Th227.n.photon.energy.out Outputs/Th227.n.photon.energy.out
	./depositionEnergy --map ../Data/MG_MC/all_maps.map --pid photon --tid H1 --pa n photon H1 H2 H3 He3 He4 > Outputs/H1_photoat.n.photon.CPs.energy.out
	$(PYTHON) ../Utilities/diff.py "depositionData/depositionEnergy H1 --pa n photon CPs" Benchmarks/H1_photoat.n.photon.CPs.energy.out Outputs/H1_photoat.n.photon.CPs.energy.out

	./depositionMomentum --tid O16 > Outputs/O16.momentum.out
	$(PYTHON) ../Utilities/diff.py "depositionData/depositionMomentum O16" Benchmarks/O16.momentum.out Outputs/O16.momentum.out
	./depositionMomentum --tid O16 n > Outputs/O16.n.momentum.out
	$(PYTHON) ../Utilities/diff.py "depositionData/depositionMomentum O16 n" Benchmarks/O16.n.momentum.out Outputs/O16.n.momentum.out
	./depositionMomentum --tid O16 n photon > Outputs/O16.n.photon.momentum.out
	$(PYTHON) ../Utilities/diff.py "depositionData/depositionMomentum O16 n photon" Benchmarks/O16.n.photon.momentum.out Outputs/O16.n.photon.momentum.out
	./depositionMomentum --tid Th227 > Outputs/Th227.momentum.out
	$(PYTHON) ../Utilities/diff.py "depositionData/depositionMomentum Th227" Benchmarks/Th227.momentum.out Outputs/Th227.momentum.out
	./depositionMomentum --tid Th227 n > Outputs/Th227.n.momentum.out
	$(PYTHON) ../Utilities/diff.py "depositionData/depositionMomentum Th227 n" Benchmarks/Th227.n.momentum.out Outputs/Th227.n.momentum.out
	./depositionMomentum --tid Th227 n photon > Outputs/Th227.n.photon.momentum.out
	$(PYTHON) ../Utilities/diff.py "depositionData/depositionMomentum Th227 n photon" Benchmarks/Th227.n.photon.momentum.out Outputs/Th227.n.photon.momentum.out
	./depositionMomentum --map ../Data/MG_MC/all_maps.map --pid photon --tid H1 --pa n photon H1 H2 H3 He3 He4 > Outputs/H1_photoat.n.photon.CPs.momentum.out
	$(PYTHON) ../Utilities/diff.py "depositionData/depositionEnergy H1 --pa n photon CPs" Benchmarks/H1_photoat.n.photon.CPs.momentum.out Outputs/H1_photoat.n.photon.CPs.momentum.out

	./photo-atomic photon > Outputs/photo-atomic.photon
	$(PYTHON) ../Utilities/diff.py "depositionData/photo-atomic photon" Benchmarks/photo-atomic.photon Outputs/photo-atomic.photon
