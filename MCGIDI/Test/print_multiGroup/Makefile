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

check: print_multiGroup
	if [ ! -e Outputs ]; then mkdir Outputs; fi
	./print_multiGroup > Outputs/print_multiGroup.out
	../Utilities/diff.com print_multiGroup/print_multiGroup Benchmarks/print_multiGroup.out Outputs/print_multiGroup.out
	./print_multiGroup --gid LLNL_gid_93 > Outputs/print_multiGroup--gid_LLNL_gid_93.out
	../Utilities/diff.com print_multiGroup/print_multiGroup Benchmarks/print_multiGroup--gid_LLNL_gid_93.out Outputs/print_multiGroup--gid_LLNL_gid_93.out
