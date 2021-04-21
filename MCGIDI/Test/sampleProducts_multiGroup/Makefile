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

check: sampleProducts_multiGroup
	python sampleProducts_multiGroup.py sampleProducts_multiGroup --all
	python sampleProducts_multiGroup.py sampleProducts_multiGroup --tid Th227 --all
	python sampleProducts_multiGroup.py sampleProducts_multiGroup --tid Th227 -d --all
	python sampleProducts_multiGroup.py sampleProducts_multiGroup --tid Th227 -d
	python sampleProducts_multiGroup.py sampleProducts_multiGroup --tid Th227 -d -p
	python sampleProducts_multiGroup.py sampleProducts_multiGroup --tid Th227 -p

	python sampleProducts_multiGroup.py sampleProducts_multiGroup --pid photon --tid O16 --map ../../../GIDI/Test/Data/MG_MC/all.map --pa
	python sampleProducts_multiGroup.py sampleProducts_multiGroup --pid photon --tid O16 --map ../../../GIDI/Test/Data/MG_MC/all.map --pa -n
	python sampleProducts_multiGroup.py sampleProducts_multiGroup --pid photon --tid O16 --map ../../../GIDI/Test/Data/MG_MC/all.map --pn
	python sampleProducts_multiGroup.py sampleProducts_multiGroup --pid photon --tid O16 --map ../../../GIDI/Test/Data/MG_MC/all.map --pn -p
	python sampleProducts_multiGroup.py sampleProducts_multiGroup --pid photon --tid O16 --map ../../../GIDI/Test/Data/MG_MC/all.map --pa --pn -p -n

	python sampleProducts_multiGroup.py sampleProducts_multiGroup --tid HinCH2 --map ../../../GIDI/Test/Data/MG_MC/neutrons/all.map
