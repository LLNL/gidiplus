/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#ifndef LUPI_defines_hpp_included
#define LUPI_defines_hpp_included 1

#if __cplusplus > 201402L
    #define LUPI_maybeUnused [[maybe_unused]]
#else
    #define LUPI_maybeUnused 
#endif

#endif          // LUPI_defines_hpp_included
