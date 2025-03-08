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
	-./productIndices > Outputs/productIndices.out
	-../Utilities/diff.com productIndices/productIndices Benchmarks/productIndices.out Outputs/productIndices.out
	-./productMultiplicities > Outputs/productMultiplicities.out
	-../Utilities/diff.com productIndices/productMultiplicities Benchmarks/productMultiplicities.out Outputs/productMultiplicities.out
	-./productAverageMultiplicities > Outputs/productAverageMultiplicities.out
	-../Utilities/diffSkipFirstLine.com productIndices/productAverageMultiplicities Benchmarks/productAverageMultiplicities.out Outputs/productAverageMultiplicities.out

	-./productIndices --tid Th227 > Outputs/productIndices--tid.Th227.out
	-../Utilities/diffSkipFirstLine.com productIndices/productIndices Benchmarks/productIndices--tid.Th227.out Outputs/productIndices--tid.Th227.out
	-./productIndices --tid Th227 --ENDL99120 > Outputs/productIndices--tid.Th227.ENDL99120.out
	-../Utilities/diffSkipFirstLine.com productIndices/productIndices Benchmarks/productIndices--tid.Th227.ENDL99120.out Outputs/productIndices--tid.Th227.ENDL99120.out

	-./productMultiplicities --tid Th227 > Outputs/productMultiplicities--tid.Th227.out
	-../Utilities/diffSkipFirstLine.com productIndices/productMultiplicities Benchmarks/productMultiplicities--tid.Th227.out Outputs/productMultiplicities--tid.Th227.out
	-./productMultiplicities --tid Th227 --ENDL99120 > Outputs/productMultiplicities--tid.Th227.ENDL99120.out
	-../Utilities/diffSkipFirstLine.com productIndices/productMultiplicities Benchmarks/productMultiplicities--tid.Th227.ENDL99120.out Outputs/productMultiplicities--tid.Th227.ENDL99120.out

	-./productAverageMultiplicities --tid Th227 > Outputs/productAverageMultiplicities--tid.Th227.out
	-../Utilities/diffSkipFirstLine.com productIndices/productAverageMultiplicities Benchmarks/productAverageMultiplicities--tid.Th227.out Outputs/productAverageMultiplicities--tid.Th227.out

	-./productAverageMultiplicities -d --tid Th227 > Outputs/productAverageMultiplicities--tid.Th227-d.out
	-../Utilities/diffSkipFirstLine.com productIndices/productAverageMultiplicities Benchmarks/productAverageMultiplicities--tid.Th227-d.out Outputs/productAverageMultiplicities--tid.Th227-d.out
	-./productAverageMultiplicities -d --tid Th227 --ENDL99120 > Outputs/productAverageMultiplicities--tid.Th227.ENDL99120-d.out
	-../Utilities/diffSkipFirstLine.com productIndices/productAverageMultiplicities Benchmarks/productAverageMultiplicities--tid.Th227.ENDL99120-d.out Outputs/productAverageMultiplicities--tid.Th227.ENDL99120-d.out
