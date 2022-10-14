SHELL = /bin/ksh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

.PHONY: default bin compile check clean realclean doDirs

DIRS = Src

default: compile

bin:
	$(MAKE) doDirs TARGET=bin

compile:
	$(MAKE) doDirs TARGET=default

check:

clean: realclean

realclean:
	$(MAKE) doDirs TARGET=realclean
	rm -f readAllProtaresInMapFile directory toXML reactions GNDS_FileType incompleteParticles HAPI_ReadAllProtaresInMapFile mapTiming \
		multiGroupCrossSections producedPaticles readAllProtaresInMapFilePrintMultiGroup

doDirs:
	for dir in $(DIRS); do cd $$dir; $(MAKE) $(TARGET); cd ../; done
