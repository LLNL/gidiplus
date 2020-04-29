SHELL = /bin/ksh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

.PHONY: default clean realclean

localCXXFLAGS = $(CXXFLAGS) -I../../pugixml/include

CppSources = PoPI_database.cpp PoPI_base.cpp PoPI_particle.cpp PoPI_gaugeBoson.cpp PoPI_lepton.cpp PoPI_baryon.cpp \
    PoPI_chemicalElement.cpp PoPI_isotope.cpp PoPI_nuclide.cpp PoPI_nucleus.cpp \
    PoPI_unorthodox.cpp PoPI_alias.cpp \
    PoPI_pq_suite.cpp PoPI_physicalQuantity.cpp \
    PoPI_decayData.cpp PoPI_misc.cpp PoPI_nuclideGammaBranching.cpp

CppObjs = $(CppSources:.cpp=.o)

default: $(CppObjs)
	rm -f libPoPI.a
	$(AR) -cr libPoPI.a *.o

%.o: %.cpp
	echo "    "$<
	$(CXX) -c $(localCXXFLAGS) $< -o $@

clean:
	rm -f *.o *.so *.a

realclean: clean
