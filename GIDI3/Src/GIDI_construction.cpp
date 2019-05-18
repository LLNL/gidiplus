/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include <stdlib.h>
#include <string.h>
#include <limits.h> 

#include "GIDI.hpp"

namespace GIDI {

namespace Construction {

/*! \class Settings
 * This class is used to pass user parameters to various constructors.
 *
 * The main use is to limit the type of data read in via the **a_parseMode** argument (see enum ParseMode).
*/

/* *********************************************************************************************************//**
 * @param a_parseMode           [in]    Instructs the parses on which data to parse.
 * @param a_photoMode           [in]    Instructs the parses if photo atomic and/or photoicnuclear protares are to be included.
 ***********************************************************************************************************/

Settings::Settings( ParseMode a_parseMode, PhotoMode a_photoMode ) :
        m_parseMode( a_parseMode ),
        m_photoMode( a_photoMode ),
        m_useSystem_strtod( 0 ) {

};

}

}
