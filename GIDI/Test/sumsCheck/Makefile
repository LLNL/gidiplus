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
	./sumsCheck > Outputs/sumsCheck.out
	../Utilities/diff.com sumsCheck/sumsCheck Benchmarks/sumsCheck.out Outputs/sumsCheck.out
	./sumsCheck --tid Th227 > Outputs/sumsCheck--tid.Th227.out
	../Utilities/diff.com sumsCheck/sumsCheck Benchmarks/sumsCheck--tid.Th227.out Outputs/sumsCheck--tid.Th227.out
	./sumsCheck --pid photon --tid Th > Outputs/sumsCheck--pid.photon--tid.Th.out
	../Utilities/diff.com sumsCheck/sumsCheck Benchmarks/sumsCheck--pid.photon--tid.Th.out Outputs/sumsCheck--pid.photon--tid.Th.out
