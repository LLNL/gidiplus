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
	./regions1dAsXYs1d Data/regions1d.xml  1e-7 > Outputs/regions1d.xml.1e-7.out
	../Utilities/diff.com regions1dAsXYs1d/regions1dAsXYs1d Benchmarks/regions1d.xml.1e-7.out Outputs/regions1d.xml.1e-7.out
	./regions1dAsXYs1d Data/regions1d.xml  2e-7 > Outputs/regions1d.xml.2e-7.out
	../Utilities/diff.com regions1dAsXYs1d/regions1dAsXYs1d Benchmarks/regions1d.xml.2e-7.out Outputs/regions1d.xml.2e-7.out
	./regions1dAsXYs1d Data/regions1d.xml  6e-7 > Outputs/regions1d.xml.6e-7.out
	../Utilities/diff.com regions1dAsXYs1d/regions1dAsXYs1d Benchmarks/regions1d.xml.6e-7.out Outputs/regions1d.xml.6e-7.out
	./regions1dAsXYs1d Data/regions1d.xml 10e-7 > Outputs/regions1d.xml.10e-7.out
	../Utilities/diff.com regions1dAsXYs1d/regions1dAsXYs1d Benchmarks/regions1d.xml.10e-7.out Outputs/regions1d.xml.10e-7.out

	./regions1dAsXYs1d.2 Data/regions1d.2.lin-lin.xml 1e-4 > Outputs/regions1d.2.lin-lin.1e-4.out
	../Utilities/diff.com regions1dAsXYs1d/regions1d.2.lin-lin.1e-4.out Benchmarks/regions1d.2.lin-lin.1e-4.out Outputs/regions1d.2.lin-lin.1e-4.out
	./regions1dAsXYs1d.2 Data/regions1d.2.flat.xml 1e-4 > Outputs/regions1d.2.flat.1e-4.out
	../Utilities/diff.com regions1dAsXYs1d/regions1d.2.flat.1e-4.out Benchmarks/regions1d.2.flat.1e-4.out Outputs/regions1d.2.flat.1e-4.out
	./regions1dAsXYs1d.2 Data/regions1d.2.log-log.xml 1e-4 > Outputs/regions1d.2.log-log.1e-4.out
	../Utilities/diff.com regions1dAsXYs1d/regions1d.2.log-log.1e-4.out Benchmarks/regions1d.2.log-log.1e-4.out Outputs/regions1d.2.log-log.1e-4.out

	./curve1d Data/constand1d.xml 1e-3 > Outputs/constand1d.out
	../Utilities/diff.com regions1dAsXYs1d/constand1d.out Benchmarks/constand1d.out Outputs/constand1d.out

	./curve1d Data/Legendre1d.xml 1e-3 > Outputs/Legendre1d.out
	../Utilities/diff.com regions1dAsXYs1d/Legendre1d.out Benchmarks/Legendre1d.out Outputs/Legendre1d.out

	./curve1d Data/polynomial1d.xml 1e-3 > Outputs/polynomial1d.out
	../Utilities/diff.com regions1dAsXYs1d/polynomial1d.out Benchmarks/polynomial1d.out Outputs/polynomial1d.out
