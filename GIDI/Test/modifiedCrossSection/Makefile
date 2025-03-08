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
	./modifiedCrossSection.sh modifiedCrossSection
	./modifiedCrossSection.sh modifiedCrossSection___0.2_ --offset 0.2
	./modifiedCrossSection.sh modifiedCrossSection_1e-3_2_0.2_ --offset 0.2 --domainMin 1e-3 --domainMax 2
	./modifiedCrossSection.sh modifiedCrossSection____2 --slope 2
	./modifiedCrossSection.sh modifiedCrossSection_1e-3_2__2 --slope 2 --domainMin 1e-3 --domainMax 2
	./modifiedCrossSection.sh modifiedCrossSection_1e-3_2_0.2_2 --offset 0.2 --slope 2 --domainMin 1e-3 --domainMax 2
	./modifiedCrossSection.sh modifiedCrossSection_1e-3_150_0.2_2 --offset 0.2 --slope 2 --domainMin 1e-3 --domainMax 150

	./modifiedCrossSection_multiGroup.sh modifiedCrossSection_multiGroup
	./modifiedCrossSection_multiGroup.sh modifiedCrossSection_multiGroup____2 --slope 2
	./modifiedCrossSection_multiGroup.sh modifiedCrossSection_multiGroup_1e-3_2_0.2_ --offset 0.2 --domainMin 1e-3 --domainMax 2
	./modifiedCrossSection_multiGroup.sh modifiedCrossSection_multiGroup_1e-3_2_0.2_2 --offset 0.2 --slope 2 --domainMin 1e-3 --domainMax 2
