# GIDI
GIDI (General Interaction Data Interface) is a C++ library for reading and writing nuclear reaction data stored in the GNDS (Generalized Nuclear Data Structure) format

# Installation
See the GIDIplus Readme for installation instructions

# License
GIDI is distributed under the terms of the MIT license.

SPDX-License-Identifier: MIT

# Test data
GIDI test data is not included in the repository, but is available from the 'releases' section at https://github.com/LLNL/gidiplus/releases/tag/v3.19.71.
To run `make check`, first download the test data and place it in GIDI/Test

The GIDI data files need to be extracted from the TAR compressed archive with the following command:

    `cd Test; tar -xzvf GIDI_Test_Data.tar.gz`

