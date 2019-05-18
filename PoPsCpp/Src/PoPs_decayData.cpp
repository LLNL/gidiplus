/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include "PoPs.hpp"

namespace PoPs {

#define decayModesString "decayModes"
#define decayPathString "decayPath"
#define productsString "products"

/*
============================================================
======================== DecayData =========================
============================================================
*/
DecayData::DecayData( pugi::xml_node const &a_node ) :
        m_decayModes( decayModesString ) {

    m_decayModes.appendFromParentNode2( a_node.child( "decayModes" ), this );
}
/*
=========================================================
*/
DecayData::~DecayData( ) {

}
/*
=========================================================
*/
void DecayData::calculateNuclideGammaBranchStateInfo( PoPs::Database const &a_pops, NuclideGammaBranchStateInfo &a_nuclideGammaBranchStateInfo ) const {

    for( std::size_t i1 = 0; i1 <  m_decayModes.size( ); ++i1 ) {
        DecayMode const &decayMode = m_decayModes[i1];

        decayMode.calculateNuclideGammaBranchStateInfo( a_pops, a_nuclideGammaBranchStateInfo );
    }
}
/*
=========================================================
*/
void DecayData::toXMLList( std::vector<std::string> &a_XMLList, std::string const &a_indent1 ) const {

    std::string::size_type size = m_decayModes.size( );

    if( size == 0 ) return;

    std::string header = a_indent1 + "<decayData>";
    a_XMLList.push_back( header );

    if( size > 0 ) {
        std::string indent2 = a_indent1 + "  ";
        m_decayModes.toXMLList( a_XMLList, indent2 );
    }

    appendXMLEnd( a_XMLList, "decayData" );
}

/*
============================================================
======================== DecayMode =========================
============================================================
*/
DecayMode::DecayMode( pugi::xml_node const &a_node, DecayData const *a_decayData ) :
        m_label( a_node.attribute( "label" ).value( ) ),
        m_mode( a_node.attribute( "mode" ).value( ) ),
        m_probability( a_node.child( "probability" ) ),
        m_photonEmissionProbabilities( a_node.child( "photonEmissionProbabilities" ) ),
        m_decayPath( decayPathString ) {

    m_decayPath.appendFromParentNode2( a_node.child( "decayPath" ), this );
}
/*
============================================================
*/
DecayMode::~DecayMode( ) {

}
/*
============================================================
*/
void DecayMode::calculateNuclideGammaBranchStateInfo( PoPs::Database const &a_pops, NuclideGammaBranchStateInfo &a_nuclideGammaBranchStateInfo ) const {

    if( m_mode == PoPs_decayMode_electroMagnetic ) {
        double _probability = getPhysicalQuantityOfSuiteAsDouble( probability( ) );
        double _photonEmissionProbabilities = getPhysicalQuantityOfSuiteAsDouble( photonEmissionProbabilities( ), true, 1.0 );

        std::string residualState( "" );
        Decay const &decay = m_decayPath[0];
        Suite<Product, Decay> const &products = decay.products( );
        for( std::size_t i1 = 0; i1 < products.size( ); ++i1 ) {
            Product const &product = products[i1];

            if( product.pid( ) != IDs::photon ) residualState = product.pid( );
        }

        Particle const &initialState = a_pops.get<Particle>( a_nuclideGammaBranchStateInfo.state( ) );
        Particle const &finalState = a_pops.get<Particle>( residualState );
        double gammaEnergy = PoPs_AMU2MeV_c2 * ( initialState.massValue( "amu" ) - finalState.massValue( "amu" ) );

        NuclideGammaBranchInfo nuclideGammaBranchInfo( _probability, _photonEmissionProbabilities, gammaEnergy, residualState );
        a_nuclideGammaBranchStateInfo.add( nuclideGammaBranchInfo );
    }
}
/*
============================================================
*/
void DecayMode::toXMLList( std::vector<std::string> &a_XMLList, std::string const &a_indent1 ) const {

    std::string header = a_indent1 + "<decayMode label=\"" + m_label + "\" mode=\"" + m_mode + "\">";
    a_XMLList.push_back( header );

    std::string indent2 = a_indent1 + "  ";
    m_probability.toXMLList( a_XMLList, indent2 );
    m_decayPath.toXMLList( a_XMLList, indent2 );

    appendXMLEnd( a_XMLList, "decayMode" );
}

/*
============================================================
========================== Decay ===========================
============================================================
*/
Decay::Decay( pugi::xml_node const &a_node, DecayMode const *a_decayMode ) :
        m_index( a_node.attribute( "index" ).as_int( ) ),
        m_products( productsString ) {

    m_products.appendFromParentNode2( a_node.child( "products" ), this );
}
/*
============================================================
*/
Decay::~Decay( ) {

}
/*
============================================================
*/
void Decay::toXMLList( std::vector<std::string> &a_XMLList, std::string const &a_indent1 ) const {

    char str[64];
    sprintf( str, "%d", m_index );
    std::string indexString( str );

    std::string header = a_indent1 + "<decay index=\"" + indexString + "\">";
    a_XMLList.push_back( header );

    std::string indent2 = a_indent1 + "  ";
    m_products.toXMLList( a_XMLList, indent2 );

    appendXMLEnd( a_XMLList, "decay" );

}

/*
============================================================
========================= Product ==========================
============================================================
*/
Product::Product( pugi::xml_node const &a_node, Decay *a_DB ) :
        m_id( -1 ),
        m_pid( a_node.attribute( "pid" ).value( ) ),
        m_label( a_node.attribute( "label" ).value( ) ) {

}
/*
============================================================
*/
Product::~Product( ) {

}
/*
============================================================
*/
void Product::toXMLList( std::vector<std::string> &a_XMLList, std::string const &a_indent1 ) const {

    std::string header = a_indent1 + "<product label=\"" + m_label + "\" pid=\"" + m_pid + "\"/>";
    a_XMLList.push_back( header );
}

}
