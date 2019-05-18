# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

SHELL = /bin/sh

.PHONY: clean realclean

default:
	cd ../Src; $(MAKE)
	$(AR) -rc libstatusMessageReporting.a ../Src/statusMessageReporting.o
	which $(RANLIB) > /dev/null 2>&1; if [ $$? -eq 0 ]; then $(RANLIB) libstatusMessageReporting.a; fi

clean:
	rm -f libstatusMessageReporting.a

realclean: clean
