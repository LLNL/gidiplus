/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#ifndef GIDIP_python_hpp_included
#define GIDIP_python_hpp_included 1

#include <LUPI.hpp>

namespace GIDIP {

namespace Python {

void *loadModule( std::string const &a_moduleName );
void *loadFunctionInModule( std::string const &a_moduleName, std::string const &a_functionName );
void decrementRef( void *a_pyObject );
double callFunctionReturnDouble( void *a_PyFunc, void *a_PyArg );

}               // End of namespace Python.

}               // End of namespace GIDIP.

#endif          // GIDIP_python_hpp_included
