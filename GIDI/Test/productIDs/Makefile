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
	./productIDs > Outputs/productIDs.out
	../Utilities/diff.com productIDs/productIDs Benchmarks/productIDs.out Outputs/productIDs.out
	./productIDs --pa --pid photon > Outputs/productIDs.photon.pa.out
	../Utilities/diff.com productIDs/productIDs Benchmarks/productIDs.photon.pa.out Outputs/productIDs.photon.pa.out
	./productIDs --tid Th227 > Outputs/productIDs.Th227.out
	../Utilities/diff.com productIDs/productIDs Benchmarks/productIDs.Th227.out Outputs/productIDs.Th227.out
	./productIDs --tid Th227 --ENDL99120 > Outputs/productIDs.Th227.ENDL99120.out
	../Utilities/diff.com productIDs/productIDs Benchmarks/productIDs.Th227.ENDL99120.out Outputs/productIDs.Th227.ENDL99120.out
