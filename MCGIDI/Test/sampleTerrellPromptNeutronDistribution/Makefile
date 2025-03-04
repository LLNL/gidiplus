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

check: sampleTerrellPromptNeutronDistribution
	if [ ! -e Outputs ]; then mkdir Outputs; fi
	./sampleTerrellPromptNeutronDistribution > Outputs/sampleTerrellPromptNeutronDistribution.out
	../Utilities/diff.com sampleTerrellPromptNeutronDistribution/sampleTerrellPromptNeutronDistribution Benchmarks/sampleTerrellPromptNeutronDistribution.out Outputs/sampleTerrellPromptNeutronDistribution.out
	./sampleTerrellPromptNeutronDistribution -t > Outputs/sampleTerrellPromptNeutronDistribution_-t.out
	../Utilities/diff.com sampleTerrellPromptNeutronDistribution/sampleTerrellPromptNeutronDistribution Benchmarks/sampleTerrellPromptNeutronDistribution_-t.out Outputs/sampleTerrellPromptNeutronDistribution_-t.out
