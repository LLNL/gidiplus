SHELL = /bin/ksh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

GIDI_PLUS_PATH ?= $(abspath ../../..)
CppSource = $(sort $(wildcard *.cpp))

include ../../Makefile.paths
include ../Makefile.check

check: productIndices productMultiplicities
	if [ ! -e Outputs ]; then mkdir Outputs; fi
	-./productIndices > Outputs/productIndices.out
	-../Utilities/diff.com productIndices/productIndices Benchmarks/productIndices.out Outputs/productIndices.out
	-./productMultiplicities > Outputs/productMultiplicities.out
	-../Utilities/diff.com productIndices/productMultiplicities Benchmarks/productMultiplicities.out Outputs/productMultiplicities.out
