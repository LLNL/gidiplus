SHELL = /bin/bash

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

# These must be set by hand when we do a release.
gidiplus_major=3
gidiplus_minor=17

DIRS = pugixml statusMessageReporting PoPsCpp numericalFunctions GIDI3 include lib

PREFIX = `pwd`/install

.PHONY: default include lib pugixml pugixml_dummy install clean realclean tar doDIRS

default: pugixml gidiplus_version.h
	@echo "Info CXX      = $(CXX)"
	@echo "Info CXXFLAGS = $(CXXFLAGS)"
	@echo "Info CC       = $(CC)"
	@echo "Info CFLAGS   = $(CFLAGS)"
	@echo "Info PREFIX   = $(PREFIX)"
	$(MAKE) doDIRS _DIRS="$(DIRS)" _TARGET=default

include:
	cd include; $(MAKE)

lib:
	cd lib; $(MAKE)

pugixml:
	rm -rf pugixml pugixml-1.8
	unzip -q Misc/pugixml-1.8.zip
	ln -s pugixml-1.8 pugixml
	cd pugixml; tar -xf ../Misc/pugixml.addon.tar

pugixml_dummy:
	rm -rf pugixml-1.8 pugixml
	mkdir pugixml
	cp Misc/Makefile_dummy pugixml/Makefile


gidiplus_version.h: FORCE
	@if test -d "./.svn"; then \
		echo "#define GIDIPLUS_MAJOR ${gidiplus_major}"  > gidiplus_version.h; \
		echo "#define GIDIPLUS_MINOR ${gidiplus_minor}" >> gidiplus_version.h; \
		svnversion .                      | awk 'BEGIN{FS="/+| +"} {printf("#define GIDIPLUS_PATCHLEVEL %d\n" , $$1)}'          >> gidiplus_version.h; \
		svnversion .                      | awk 'BEGIN{FS="/+| +"} {printf("#define GIDIPLUS_SVN %d\n" , $$1)}'                 >> gidiplus_version.h; \
		svnversion statusMessageReporting | awk 'BEGIN{FS="/+| +"} {printf("#define STATUS_MESSAGE_REPORTING_SVN %d\n" , $$1)}' >> gidiplus_version.h; \
		svnversion numericalFunctions     | awk 'BEGIN{FS="/+| +"} {printf("#define NUMERICAL_FUNCTIONS_SVN  %d\n" , $$1)}'     >> gidiplus_version.h; \
		svnversion PoPsCpp                | awk 'BEGIN{FS="/+| +"} {printf("#define POPSCPP_SVN  %d\n" , $$1)}'                 >> gidiplus_version.h; \
		svnversion RCGIDI                 | awk 'BEGIN{FS="/+| +"} {printf("#define RCGIDI_SVN  %d\n" , $$1)}'                  >> gidiplus_version.h; \
	fi
	cat gidiplus_version.h

install: default
	mkdir -p $(PREFIX)/include
	cp -p include/* $(PREFIX)/include
	mkdir -p $(PREFIX)/lib
	cp -p lib/libgidiplus.a $(PREFIX)/lib

clean: FORCE 
	if [ ! -e pugixml ]; then $(MAKE) pugixml_dummy; fi
	$(MAKE) doDIRS _DIRS="$(DIRS)" _TARGET=clean

realclean: pugixml_dummy
	$(MAKE) doDIRS _DIRS="$(DIRS)" _TARGET=realclean
	rm -rf pugixml test_install

tar: gidiplus_version.h
	$(MAKE) -s realclean
	fileName=`svn info | grep Revision  | awk -F " " '{print "gidiplus-${gidiplus_major}.${gidiplus_minor}.svn"$$2}'`; rm -rf ../$$fileName; svn export -q . ../$$fileName; cp gidiplus_version.h ../$$fileName/; rm ../$$fileName/Makefile.popskit; rm ../$$fileName/*/Makefile.popskit; cd ../; tar -cf $$fileName.tar $$fileName

doDIRS:
	SAVED_PWD=`pwd`; \
	for DIR in $(_DIRS); do cd $$DIR; $(MAKE) $(_TARGET); cd $$SAVED_PWD; done

FORCE:
