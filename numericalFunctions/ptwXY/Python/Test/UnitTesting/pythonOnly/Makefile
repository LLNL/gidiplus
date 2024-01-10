# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

SHELL = /bin/sh

PYTHON = python3

.PHONY: default check checke clean realclean

TARGETS = setting.py pop.py domain.py range.py

default:

check:
	for f in $(TARGETS); do $(PYTHON) $$f; done

checke:
	$(MAKE) check CHECKOPTIONS=-e

clean:

realclean: clean
