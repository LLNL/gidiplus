# gidiplus
C++ libraries for accessing nuclear data from the Generalized Nuclear Database Structure (GNDS)

# Dependency
pugixml version 1.8, available from https://pugixml.org/2016/11/24/pugixml-1.8-release.html

# Installation
Use make to build the project (pugixml will be downloaded automatically if not present). Important targets in the Makefile are:

| Target     | Description
|------------|------------
| default:   | builds libgidiplus.a (and all other libraries) and puts them into the 'lib' directory. Puts all needed header files into the 'include' directory.
| install:   | Copies the header files from the 'include' directory to '$(PREFIX)/include'. Copies all '.a' from the 'lib' directory to '$(PREFIX)/lib.'
| realclean: | Removes all stuff built by the 'default' target.

In general, one should specify the CC and CXX compilers and their flags when building. For example,

    `make default CC=g++ CXX=g++ CXXFLAGS="-g -O3" CCFLAGS="-g -O3"`

To put the results into the path '/path/to/my/builds', execute

    `make install CC=gcc CXX=g++ CXXFLAGS="-g -O3" CCFLAGS="-g -O3" PREFIX=/path/to/my/builds`

TOSS-3 build scripts for various compilers are available in the 'Scripts' directory.

# License
gidiplus is distributed under the terms of the MIT license.

All new contributions must be made under the MIT license.

See LICENSE, COPYRIGHT and NOTICE for details.

SPDX-License-Identifier: MIT

This package includes several components
LLNL-CODE-770917	(GIDI3)
LLNL-CODE-771182	(statusMessageReporting)
LLNL-CODE-770377	(PoPsCpp)
LLNL-CODE-770134	(numericalFunctions)

