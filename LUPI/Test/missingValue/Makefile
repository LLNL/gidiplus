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
	rm -rf Outputs/*
	touch Outputs/missingValue1.out
	touch Outputs/missingValue1.err
	-./missingValue1.com ./missingValue1
	-./missingValue1.com ./missingValue1 -a a1
	-./missingValue1.com ./missingValue1 -a a1 A1 A2
	-./missingValue1.com ./missingValue1 -a a1 A1 A2 A3
	-./missingValue1.com ./missingValue1 -a a1 A1 A2 A3 A4
	-./missingValue1.com ./missingValue1 -a a1 A1 A2 A3 A4 A5
	-./missingValue1.com ./missingValue1 -a a1 A1 A2 A3 A4 -a a2
	-./missingValue1.com ./missingValue1 -a a1 A1 -a a2 A2 A3 A4
	-./missingValue1.com ./missingValue1 -a a1 A1 A2 A3 -a a2 A4
	-./missingValue1.com ./missingValue1 -a a1 A1 A2 A3 A4
	-./missingValue1.com ./missingValue1 -a a1 A1 A2 A3 -a a2
	-./missingValue1.com ./missingValue1 -a a1 A1 A2 A3 -a a2 -a a3
	-./missingValue1.com ./missingValue1 -a a1 A1 A2 A3 -a a2 -a2 a3
	-./missingValue1.com ./missingValue1 --as as1 -a a1 A1 --as as2 A2 A3 -a a2 --as
	-./missingValue1.com ./missingValue1 --as as1 -a a1 A1 --as as2 A2 A3 -a a2 --as as3 --as as4 --as as5
	-./missingValue1.com ./missingValue1 -a a1 A1 A2 A3 -a a2 --store s1
	-./missingValue1.com ./missingValue1 -a a1 A1 A2 A3 -a a2 --store s1 -s s2
	-./missingValue1.com ./missingValue1 -a a1 A1 A2 A3 -a a2 --store s1 -s s2 --store s3
	-./missingValue1.com ./missingValue1 -a a1 A1 A2 A3 -a a2 --store s1 -s s2 --store s3 -s s4
	-./missingValue1.com ./missingValue1 -a a1 A1 A2 A3 -a a2 --store s1 -s s2 --store s3 -s
	-diff -r Benchmarks Outputs > /dev/null; if [ $$? -ne 0 ]; then echo "FAILURE: missingValue"; fi
