# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

SHELL = /bin/sh

PYTHON = python3

.PHONY: default check clean realclean

default:

check:
	$(PYTHON) thin.py
	$(PYTHON) thinDomain.py

clean:

realclean: clean
