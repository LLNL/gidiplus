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
	touch Outputs/alias.out
	touch Outputs/alias.err
	-./alias.com ./alias
	-./alias.com ./alias --true -t -y -yes --yeap
	-./alias.com ./alias --false --false
	-./alias.com ./alias --veryVeryLongCounterNameVeryVeryLong
	-./alias.com ./alias -v -v -v -v -v --veryVeryLongCounterNameVeryVeryLong -v -v
	-./alias.com ./alias --store S1
	-./alias.com ./alias --store S1 --store s1
	-./alias.com ./alias --pops p1.xml -p p2.xml --pop p3.xml
	-diff -r Benchmarks Outputs > /dev/null; if [ $$? -ne 0 ]; then echo "FAILURE: alias"; fi
