SHELL = /bin/ksh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

.PHONY: default clean realclean

PIGI_PATH = ../../pugixml
PIGI_INCLUDE = $(PIGI_PATH)/include

POPs_PATH = ../../PoPsCpp
POPs_INCLUDE = $(POPs_PATH)/include

SMR_PATH = ../../statusMessageReporting
SMR_INCLUDE = $(SMR_PATH)/include

NF_PATH = ../../numericalFunctions
NF_INCLUDE = $(NF_PATH)/include

local_CXXFLAGS = $(CXXFLAGS) -I$(PIGI_INCLUDE) -I$(POPs_INCLUDE) -I$(SMR_INCLUDE) -I$(NF_INCLUDE)

CppSources = GIDI_map.cpp GIDI_protare.cpp GIDI_protareComposite.cpp GIDI_protareTNSL.cpp \
        GIDI_reaction.cpp GIDI_outputChannel.cpp GIDI_product.cpp \
        GIDI_fissionFragmentData.cpp GIDI_delayedNeutron.cpp \
		GIDI_sums.cpp \
		GIDI_distributions.cpp \
		GIDI_ancestry.cpp GIDI_suite.cpp GIDI_form.cpp GIDI_particleInfo.cpp \
		GIDI_documentation.cpp \
		GIDI_styles.cpp \
		GIDI_flux.cpp \
		GIDI_transportable.cpp \
		GIDI_group.cpp \
		GIDI_vector.cpp GIDI_matrix.cpp GIDI_array3d.cpp \
		GIDI_axis.cpp GIDI_grid.cpp GIDI_axes.cpp \
		GIDI_1dData.cpp GIDI_constant1d.cpp GIDI_XYs1d.cpp GIDI_Ys1d.cpp GIDI_gridded1d.cpp GIDI_xs_pdf_cdf1d.cpp GIDI_regions1d.cpp GIDI_reference1d.cpp \
		GIDI_polynomial1d.cpp GIDI_Legendre1d.cpp GIDI_branching1d.cpp GIDI_unspecified1d.cpp GIDI_misc1d.cpp \
        GIDI_resonancesWithBackground1d.cpp GIDI_URR_probabilityTables1d.cpp \
		GIDI_XYs2d.cpp GIDI_recoil2d.cpp GIDI_isotropic2d.cpp GIDI_regions2d.cpp GIDI_discreteGamma2d.cpp GIDI_primaryGamma2d.cpp \
		GIDI_energyFunctions2d.cpp \
		GIDI_XYs3d.cpp GIDI_gridded3d.cpp \
		GIDI_doubleDifferentialCrossSection.cpp \
		GIDI_arrays.cpp \
		GIDI_physicalQuantity.cpp GIDI_axisDomain.cpp \
		GIDI_parseSuites.cpp \
		GIDI_fissionEnergyReleased.cpp \
		GIDI_settings.cpp GIDI_settings_particle.cpp GIDI_settings_group.cpp GIDI_settings_flux.cpp \
		GIDI_collapse.cpp \
		GIDI_misc.cpp \
		GIDI_construction.cpp

CppObjs = $(CppSources:.cpp=.o)

default: $(CppObjs)
	rm -f libGIDI.a
	$(AR) -cr libGIDI.a *.o

%.o: %.cpp
	echo $<
	$(CXX) -c $(local_CXXFLAGS) $< -o $@

GIDI.hpp: GIDI_data.hpp

GIDI_protare.o : GIDI_protare.cpp GIDI.hpp
GIDI_protareComposite.o : GIDI_protareComposite.cpp GIDI.hpp
GIDI_reaction.o : GIDI_reaction.cpp GIDI.hpp
GIDI_outputChannel.o : GIDI_outputChannel.cpp GIDI.hpp
GIDI_product.o : GIDI_product.cpp GIDI.hpp
GIDI_suite.o : GIDI_suite.cpp GIDI.hpp
GIDI_form.cpp : GIDI.hpp
GIDI_styles.cpp : GIDI.hpp
GIDI_vector.cpp : GIDI.hpp
GIDI_matrix.cpp : GIDI.hpp
GIDI_axis.cpp : GIDI.hpp
GIDI_grid.cpp : GIDI.hpp
GIDI_axes.cpp : GIDI.hpp
GIDI_1dData.cpp : GIDI.hpp
GIDI_gridded1d.cpp : GIDI.hpp
GIDI_arrays.cpp : GIDI.hpp
GIDI_physicalQuantity.cpp : GIDI.hpp
GIDI_collapse.cpp : GIDI.hpp

clean:
	rm -f *.o *.so *.a

realclean: clean
