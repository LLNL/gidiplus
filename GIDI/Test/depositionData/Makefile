SHELL = /bin/ksh

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
	python ../Utilities/diff.py "depositionData/depositionEnergy O16" Benchmarks/O16.energy.out Outputs/O16.energy.out
	./depositionEnergy --tid O16 n > Outputs/O16.n.energy.out
	python ../Utilities/diff.py "depositionData/depositionEnergy O16 n" Benchmarks/O16.n.energy.out Outputs/O16.n.energy.out
	./depositionEnergy --tid O16 n photon > Outputs/O16.n.photon.energy.out 
	python ../Utilities/diff.py "depositionData/depositionEnergy O16 n photon" Benchmarks/O16.n.photon.energy.out Outputs/O16.n.photon.energy.out
	./depositionEnergy --tid Th227 > Outputs/Th227.energy.out
	python ../Utilities/diff.py "depositionData/depositionEnergy Th227" Benchmarks/Th227.energy.out Outputs/Th227.energy.out
	./depositionEnergy --tid Th227 n > Outputs/Th227.n.energy.out
	python ../Utilities/diff.py "depositionData/depositionEnergy Th227 n" Benchmarks/Th227.n.energy.out Outputs/Th227.n.energy.out
	./depositionEnergy --tid Th227 n photon > Outputs/Th227.n.photon.energy.out
	python ../Utilities/diff.py "depositionData/depositionEnergy Th227 n photon" Benchmarks/Th227.n.photon.energy.out Outputs/Th227.n.photon.energy.out

	./depositionMomentum --tid O16 > Outputs/O16.momentum.out
	python ../Utilities/diff.py "depositionData/depositionMomentum O16" Benchmarks/O16.momentum.out Outputs/O16.momentum.out
	./depositionMomentum --tid O16 n > Outputs/O16.n.momentum.out
	python ../Utilities/diff.py "depositionData/depositionMomentum O16 n" Benchmarks/O16.n.momentum.out Outputs/O16.n.momentum.out
	./depositionMomentum --tid O16 n photon > Outputs/O16.n.photon.momentum.out
	python ../Utilities/diff.py "depositionData/depositionMomentum O16 n photon" Benchmarks/O16.n.photon.momentum.out Outputs/O16.n.photon.momentum.out
	./depositionMomentum --tid Th227 > Outputs/Th227.momentum.out
	python ../Utilities/diff.py "depositionData/depositionMomentum Th227" Benchmarks/Th227.momentum.out Outputs/Th227.momentum.out
	./depositionMomentum --tid Th227 n > Outputs/Th227.n.momentum.out
	python ../Utilities/diff.py "depositionData/depositionMomentum Th227 n" Benchmarks/Th227.n.momentum.out Outputs/Th227.n.momentum.out
	./depositionMomentum --tid Th227 n photon > Outputs/Th227.n.photon.momentum.out
	python ../Utilities/diff.py "depositionData/depositionMomentum Th227 n photon" Benchmarks/Th227.n.photon.momentum.out Outputs/Th227.n.photon.momentum.out
