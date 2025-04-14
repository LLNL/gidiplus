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

check: sampleProducts_multiGroup
	$(PYTHON) sampleProducts_multiGroup.py sampleProducts_multiGroup --all
	$(PYTHON) sampleProducts_multiGroup.py sampleProducts_multiGroup --tid Th227 --all
	$(PYTHON) sampleProducts_multiGroup.py sampleProducts_multiGroup --tid Th227 -d --all
	$(PYTHON) sampleProducts_multiGroup.py sampleProducts_multiGroup --tid Th227 -d
	$(PYTHON) sampleProducts_multiGroup.py sampleProducts_multiGroup --tid Th227 -d -p
	$(PYTHON) sampleProducts_multiGroup.py sampleProducts_multiGroup --tid Th227 -p

	$(PYTHON) sampleProducts_multiGroup.py sampleProducts_multiGroup --pid photon --tid O16 --map ../../../GIDI/Test/Data/MG_MC/all.map --pa
	$(PYTHON) sampleProducts_multiGroup.py sampleProducts_multiGroup --pid photon --tid O16 --map ../../../GIDI/Test/Data/MG_MC/all.map --pa -n
	$(PYTHON) sampleProducts_multiGroup.py sampleProducts_multiGroup --pid photon --tid O16 --map ../../../GIDI/Test/Data/MG_MC/all.map --pn
	$(PYTHON) sampleProducts_multiGroup.py sampleProducts_multiGroup --pid photon --tid O16 --map ../../../GIDI/Test/Data/MG_MC/all.map --pn -p
	$(PYTHON) sampleProducts_multiGroup.py sampleProducts_multiGroup --pid photon --tid O16 --map ../../../GIDI/Test/Data/MG_MC/all.map --pa --pn -p -n

	$(PYTHON) sampleProducts_multiGroup.py sampleProducts_multiGroup --tid HinCH2 --map ../../../GIDI/Test/Data/MG_MC/neutrons/all.map
