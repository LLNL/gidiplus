SHELL = /bin/ksh

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

check: sampleProducts
	if [ ! -e Outputs ]; then mkdir Outputs; fi
	./sampleProducts --all > Outputs/sampleProducts.out
	python diffKE.py sampleProducts/sampleProducts Benchmarks/sampleProducts.out Outputs/sampleProducts.out

	./sampleProducts --tid Th227 --all > Outputs/sampleProducts.Th227.noDelayed.out
	python diffKE.py sampleProducts/sampleProducts Benchmarks/sampleProducts.Th227.noDelayed.out Outputs/sampleProducts.Th227.noDelayed.out
	./sampleProducts --tid Th227 -d --all > Outputs/sampleProducts.Th227.Delayed.out
	python diffKE.py sampleProducts/sampleProducts Benchmarks/sampleProducts.Th227.Delayed.out Outputs/sampleProducts.Th227.Delayed.out
	./sampleProducts --tid Th227 -d > Outputs/sampleProducts.Th227.n.Delayed.out
	python diffKE.py sampleProducts/sampleProducts Benchmarks/sampleProducts.Th227.n.Delayed.out Outputs/sampleProducts.Th227.n.Delayed.out
	./sampleProducts --tid Th227 -d -p > Outputs/sampleProducts.Th227.np.Delayed.out
	python diffKE.py sampleProducts/sampleProducts Benchmarks/sampleProducts.Th227.np.Delayed.out Outputs/sampleProducts.Th227.np.Delayed.out
	./sampleProducts --tid Th227 -p > Outputs/sampleProducts.Th227.np.noDelayed.out
	python diffKE.py sampleProducts/sampleProducts Benchmarks/sampleProducts.Th227.np.noDelayed.out Outputs/sampleProducts.Th227.np.noDelayed.out

	./sampleProducts --pid photon --tid O16 --map ../../../GIDI/Test/Data/MG_MC/all.map --pa > Outputs/sampleProducts.photon.pa.out
	python diffKE.py sampleProducts/sampleProducts Benchmarks/sampleProducts.photon.pa.out Outputs/sampleProducts.photon.pa.out
	./sampleProducts --pid photon --tid O16 --map ../../../GIDI/Test/Data/MG_MC/all.map --pa -n > Outputs/sampleProducts.photon.pa.out
	python diffKE.py sampleProducts/sampleProducts Benchmarks/sampleProducts.photon.pa.out Outputs/sampleProducts.photon.pa.out
	./sampleProducts --pid photon --tid O16 --map ../../../GIDI/Test/Data/MG_MC/all.map --pn > Outputs/sampleProducts.photon.pn.out
	python diffKE.py sampleProducts/sampleProducts Benchmarks/sampleProducts.photon.pn.out Outputs/sampleProducts.photon.pn.out
	./sampleProducts --pid photon --tid O16 --map ../../../GIDI/Test/Data/MG_MC/all.map --pn -p > Outputs/sampleProducts.photon.pn.photon.out
	python diffKE.py sampleProducts/sampleProducts Benchmarks/sampleProducts.photon.pn.photon.out Outputs/sampleProducts.photon.pn.photon.out
	./sampleProducts --pid photon --tid O16 --map ../../../GIDI/Test/Data/MG_MC/all.map --pa --pn -p -n > Outputs/sampleProducts.photon.pa.pn.neutron.photon.out
	python diffKE.py sampleProducts/sampleProducts Benchmarks/sampleProducts.photon.pa.pn.neutron.photon.out Outputs/sampleProducts.photon.pa.pn.neutron.photon.out

	./sampleProducts --tid HinCH2 --map ../../../GIDI/Test/Data/MG_MC/neutrons/all.map > Outputs/sampleProducts.HinCH2.out
	python diffKE.py sampleProducts/sampleProducts-HinCH2 Benchmarks/sampleProducts.HinCH2.out Outputs/sampleProducts.HinCH2.out
