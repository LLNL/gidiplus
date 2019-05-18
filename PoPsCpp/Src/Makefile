SHELL = /bin/ksh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

.PHONY: default clean realclean

localCXXFLAGS = $(CXXFLAGS) -I../../pugixml/include

CppSources = PoPs_database.cpp PoPs_base.cpp PoPs_particle.cpp PoPs_gaugeBoson.cpp PoPs_lepton.cpp PoPs_baryon.cpp \
    PoPs_chemicalElement.cpp PoPs_isotope.cpp PoPs_nuclide.cpp PoPs_nucleus.cpp \
    PoPs_unorthodox.cpp PoPs_alias.cpp \
    PoPs_pq_suite.cpp PoPs_physicalQuantity.cpp \
    PoPs_decayData.cpp PoPs_misc.cpp PoPs_nuclideGammaBranching.cpp

CppObjs = $(CppSources:.cpp=.o)

default: $(CppObjs)
	rm -f libPoPs.a
	$(AR) -cr libPoPs.a *.o

%.o: %.cpp
	echo $<
	$(CXX) -c $(localCXXFLAGS) $< -o $@

clean:
	rm -f *.o *.so *.a

realclean: clean
