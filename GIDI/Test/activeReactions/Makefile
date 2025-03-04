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

.PHONY: default activeReactions clean realclean

check: $(Executables)
	if [ ! -e Outputs ]; then mkdir Outputs; fi
	./activeReactions --tracking photon --tracking n --tracking photon --tracking photon               > Outputs/a
	./activeReactions --tracking photon --tracking n --tracking photon --tracking photon -r 0          > Outputs/b
	./activeReactions --tracking photon --tracking n --tracking photon --tracking photon -r 0 --invert > Outputs/c
	$(PYTHON) ./activeReactions.py activeReactions/activeReactions Outputs/a Outputs/c Outputs/b
	./activeReactions --tracking photon --tracking n --tracking photon --tracking photon -C 11 -C 40 -C 41 -C 42 -C 45 -C 48 -C 37 -C 46          > Outputs/b
	./activeReactions --tracking photon --tracking n --tracking photon --tracking photon -C 11 -C 40 -C 41 -C 42 -C 45 -C 48 -C 37 -C 46 --invert > Outputs/c
	$(PYTHON) ./activeReactions.py activeReactions/activeReactions Outputs/a Outputs/c Outputs/b
	./activeCValues -C 11 -C 12 -c 13 -c 40 -c 41 -c 11 -C 42 -c 42 > Outputs/activeCValues.out1
	../Utilities/diff.com activeReactions/Utilities Benchmarks/activeCValues.out1 Outputs/activeCValues.out1
	./activeCValues -C 11 -C 12 -c 13 -c 40 -c 41 -c 11 -C 42 -c 42 > Outputs/activeCValues.out2
	../Utilities/diff.com activeReactions/Utilities Benchmarks/activeCValues.out2 Outputs/activeCValues.out2
