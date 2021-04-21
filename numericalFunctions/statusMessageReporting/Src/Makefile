# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

SHELL = /bin/sh

localCFLAGS = $(CFLAGS) 

.PHONY: clean realclean

statusMessageReporting.o: statusMessageReporting.c statusMessageReporting.h
	$(CC) $(localCFLAGS) -I. -c -o statusMessageReporting.o statusMessageReporting.c

clean:
	rm -rf *.o

realclean: clean
