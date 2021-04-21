/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include "PoPI.hpp"

namespace PoPI {

#define PoPI_A_Chars "A"
#define PoPI_isotopeChars "isotope"

/*
=========================================================
*/
Isotope::Isotope( pugi::xml_node const &a_node, Database *a_DB, ChemicalElement *a_chemicalElement ) :
        SymbolBase( a_node, Particle_class::isotope ),
        m_chemicalElement( a_chemicalElement ),
        m_Z( a_chemicalElement->Z( ) ),
        m_A( a_node.attribute( PoPI_A_Chars ).as_int( ) ),
        m_nuclides( PoPI_nuclidesChars ) {

    m_nuclides.appendFromParentNode( a_node.child( PoPI_nuclidesChars ), a_DB, this );
}
/*
=========================================================
*/
Isotope::~Isotope( ) {

}
/*
=========================================================
*/
void Isotope::calculateNuclideGammaBranchStateInfos( PoPI::Database const &a_pops, NuclideGammaBranchStateInfos &a_nuclideGammaBranchStateInfos ) const {

    for( std::size_t i1 = 0; i1 <  m_nuclides.size( ); ++i1 ) {
        Nuclide const &nuclide = m_nuclides[i1];

        nuclide.calculateNuclideGammaBranchStateInfos( a_pops, a_nuclideGammaBranchStateInfos );
    }
}
/*
=========================================================
*/
void Isotope::toXMLList( std::vector<std::string> &a_XMLList, std::string const &a_indent1 ) const {

    std::string::size_type size = m_nuclides.size( );
    char AStr[32];
    sprintf( AStr, "%d", m_A );

    std::string header = a_indent1 + "<isotope symbol=\"" + symbol( ) + "\" A=\"" + AStr + "\">";
    a_XMLList.push_back( header );

    std::string indent2 = a_indent1 + "  ";
    std::string nuclideSuite = indent2 + "<" + PoPI_nuclidesChars + ">";
    a_XMLList.push_back( nuclideSuite );

    std::string indent3 = indent2 + "  ";
    for( std::string::size_type i1 = 0; i1 < size; ++i1 ) m_nuclides[i1].toXMLList( a_XMLList, indent3 );

    appendXMLEnd( a_XMLList, PoPI_nuclidesChars );
    appendXMLEnd( a_XMLList, PoPI_isotopeChars );
}

}
