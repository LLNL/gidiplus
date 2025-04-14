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
	./specialParticleID > Outputs/specialParticleID.out
	diff -q Benchmarks/specialParticleID.out Outputs/specialParticleID.out > /dev/null; if [ $$? != 0 ]; then echo "specialParticleID.cpp failed"; fi
	./specialParticleMasses > Outputs/specialParticleMasses.out
	diff -q Benchmarks/specialParticleMasses.out Outputs/specialParticleMasses.out > /dev/null; if [ $$? != 0 ]; then echo "specialParticleMasses.cpp failed"; fi
