/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "PoPs.hpp"

namespace PoPs {

/*
=========================================================
*/
PhysicalQuantity::PhysicalQuantity( pugi::xml_node const &a_node, pq_class a_class ) :
        m_class( a_class ),
        m_tag( a_node.name( ) ),
        m_label( a_node.attribute( "label" ).value( ) ),
        m_valueString( a_node.attribute( "value" ).value( ) ),
        m_unit( a_node.attribute( "unit" ).value( ) ) {

}
/*
=========================================================
*/
PhysicalQuantity::~PhysicalQuantity( ) {

}
/*
=========================================================
*/
void PhysicalQuantity::toXMLList( std::vector<std::string> &a_XMLList, std::string const &a_indent1 ) const {

    std::string _unit;

    if( m_unit.size( ) > 0 ) _unit = "\" unit=\"" + m_unit;

    std::string header = a_indent1 + "<" + m_tag + " label=\"" + m_label + "\" value=\"" + valueToString( ) + _unit + "\"/>";
    a_XMLList.push_back( header );
}

/*
=========================================================
*/
PQ_double::PQ_double( pugi::xml_node const &a_node ) :
        PhysicalQuantity( a_node, class_double ),
        m_value( 0.0 ) {

        initialize( );
}
/*
=========================================================
*/
PQ_double::PQ_double( pugi::xml_node const &a_node, pq_class a_class ) :
        PhysicalQuantity( a_node, a_class ),
        m_value( 0.0 ) {

        initialize( );
}
/*
=========================================================
*/
void PQ_double::initialize( ) {

    char *last;

    if( valueString( ) != "" ) m_value = strtod( valueString( ).c_str( ), &last );
}
/*
=========================================================
*/
PQ_double::~PQ_double( ) {

}
/*
=========================================================
*/
double PQ_double::value( char const *a_unit ) const {

    return( m_value );
}
/*
=========================================================
*/
std::string PQ_double::valueToString( void ) const {

    char str[64];

    sprintf( str, "%.12g", m_value );
    if( fabs( m_value ) < 1e10 ) {
        if( strchr( str, '.' ) == NULL ) sprintf( str, "%.1f", m_value );
    }
    std::string sValue( str );
    return( sValue );
}

/*
=========================================================
*/
PQ_integer::PQ_integer( pugi::xml_node const &a_node ) :
        PhysicalQuantity( a_node, class_integer ),
        m_value( a_node.attribute( "value" ).as_int( ) ) {
}
/*
=========================================================
*/
PQ_integer::~PQ_integer( ) {

}
/*
=========================================================
*/
int PQ_integer::value( char const *a_unit ) const {

    return( m_value );
}
/*
=========================================================
*/
std::string PQ_integer::valueToString( void ) const {

    char str[64];
    sprintf( str, "%d", m_value );
    std::string sValue( str );
    return( sValue );
}

/*
=========================================================
*/
PQ_fraction::PQ_fraction( pugi::xml_node const &a_node ) :
        PhysicalQuantity( a_node, class_fraction ) {

}
/*
=========================================================
*/
PQ_fraction::~PQ_fraction( ) {

}
/*
=========================================================
*/
std::string PQ_fraction::value( char const *a_unit ) const {

    return( valueString( ) );
}
/*
=========================================================
*/
std::string PQ_fraction::valueToString( void ) const {

    return( valueString( ) );
}

/*
=========================================================
*/
PQ_string::PQ_string( pugi::xml_node const &a_node ) :
        PhysicalQuantity( a_node, class_string ) {

}
/*
=========================================================
*/
PQ_string::~PQ_string( ) {

}
/*
=========================================================
*/
std::string PQ_string::value( char const *a_unit ) const {

    return( valueString( ) );
}
/*
=========================================================
*/
std::string PQ_string::valueToString( void ) const {

    return( valueString( ) );
}

/*
=========================================================
*/
PQ_shell::PQ_shell( pugi::xml_node const &a_node ) :
        PQ_double( a_node, class_shell ) {

}
/*
=========================================================
*/
PQ_shell::~PQ_shell( ) {

}

}
