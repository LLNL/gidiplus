# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

SHELL = /bin/sh

.PHONY: default headers clean realclean

default: headers

headers:
	cp $(NFUPDATING_PATH)/include/*.h .

clean:
	rm -f *.h

realclean: clean
