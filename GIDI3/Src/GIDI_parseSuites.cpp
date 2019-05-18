/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include "GIDI.hpp"

namespace GIDI {

/* *********************************************************************************************************//**
 * Function that parses a <**style**> node. Called from a Suite::parse instance.
 *
 * @param a_construction            [in]    Used to pass user options for parsing.
 * @param a_parent                  [in]    The parent GIDI::Suite that the returned Form will be added to.
 * @param a_node                    [in]    The **pugi::xml_node** to be parsed.
 * @param a_pops                    [in]    A PoPs Database instance used to get particle indices and possibly other particle information.
 * @param a_internalPoPs            [in]    The *internal* PoPs::Database instance used to get particle indices and possibly other particle information.
 *                                          This is the <**PoPs**> node under the <**reactionSuite**> node.
 * @param a_name                    [in]    The moniker for the node to be parsed.
 * @param a_styles                  [in]    A pointer to the <**styles**> node.
 * @return                                  The parsed and constructed GIDI::Form or NULL if the node is not supported.
 ***********************************************************************************************************/

Form *parseStylesSuite( Construction::Settings const &a_construction, GIDI::Suite *a_parent, pugi::xml_node const &a_node, PoPs::Database const &a_pops, 
                PoPs::Database const &a_internalPoPs, std::string const &a_name, Styles::Suite const *a_styles ) {

    Form *form = NULL;

//  Styles not parsed are angularDistributionReconstructed.

    if(      a_name == evaluatedStyleMoniker ) {
        form = new Styles::Evaluated( a_node, a_parent ); }
    else if( a_name == crossSectionReconstructedStyleMoniker ) {
        form = new Styles::CrossSectionReconstructed( a_node, a_parent ); }
    else if( a_name == CoulombPlusNuclearElasticMuCutoffStyleMoniker ) {
        form = new Styles::CoulombPlusNuclearElasticMuCutoff( a_node, a_parent ); }
    else if( a_name == TNSLStyleMoniker ) {
        form = new Styles::TNSL( a_node, a_parent ); }
    else if( a_name == averageProductDataStyleMoniker ) {
        form = new Styles::AverageProductData( a_node, a_parent ); }
    else if( a_name == MonteCarlo_cdfStyleMoniker ) {
        form = new Styles::MonteCarlo_cdf( a_node, a_parent ); }
    else if( a_name == multiGroupStyleMoniker ) {
        form = new Styles::MultiGroup( a_construction, a_node, a_pops, a_internalPoPs, a_parent ); }
    else if( a_name == heatedStyleMoniker ) {
        form = new Styles::Heated( a_node, a_parent ); }
    else if( a_name == heatedMultiGroupStyleMoniker ) {
        form = new Styles::HeatedMultiGroup( a_construction, a_node, a_pops, a_parent ); }
    else if( a_name == SnElasticUpScatterStyleMoniker ) {
        form = new Styles::SnElasticUpScatter( a_node, a_pops, a_parent ); }
    else if( a_name == griddedCrossSectionStyleMoniker ) {
        form = new Styles::GriddedCrossSection( a_construction, a_node, a_pops, a_parent ); }
    else if( a_name == URR_probabilityTablesStyleMoniker ) {
        form = new Styles::URR_probabilityTables( a_construction, a_node, a_pops, a_parent ); }
    else {
        std::cout << "parseStylesSuite: Ignoring unsupported style = '" << a_name << "'." << std::endl;
    }

    return( form );
}

/* *********************************************************************************************************//**
 * Function that parses a <**transportable**> node. Called from a Suite::parse instance.
 *
 * @param a_construction            [in]    Used to pass user options for parsing.
 * @param a_parent                  [in]    The parent GIDI::Suite that the returned Form will be added to.
 * @param a_node                    [in]    The **pugi::xml_node** to be parsed.
 * @param a_pops                    [in]    A PoPs Database instance used to get particle indices and possibly other particle information.
 * @param a_internalPoPs            [in]    The *internal* PoPs::Database instance used to get particle indices and possibly other particle information.
 *                                          This is the <**PoPs**> node under the <**reactionSuite**> node.
 * @param a_name                    [in]    The moniker for the node to be parsed.
 * @param a_styles                  [in]    A pointer to the <**styles**> node.
 * @return                                  The parsed and constructed GIDI::Form or NULL if the node is not supported.
 ***********************************************************************************************************/

Form *parseTransportablesSuite( Construction::Settings const &a_construction, Suite *a_parent, pugi::xml_node const &a_node, PoPs::Database const &a_pops, 
                PoPs::Database const &a_internalPoPs, std::string const &a_name, Styles::Suite const *a_styles ) {

    Form *form = NULL;

    if( a_name == transportableMoniker ) {
        form = new Transportable( a_construction, a_node, a_pops, a_parent ); }
    else {
        std::cout << "parseTransportablesSuite: Ignoring unsupported Form '" << a_name << "'." << std::endl;
    }

    return( form );
}

/* *********************************************************************************************************//**
 * Function that parses a <**reaction**> node. Called from a Suite::parse instance.
 *
 * @param a_construction            [in]    Used to pass user options for parsing.
 * @param a_parent                  [in]    The parent GIDI::Suite that the returned Form will be added to.
 * @param a_node                    [in]    The **pugi::xml_node** to be parsed.
 * @param a_pops                    [in]    A PoPs Database instance used to get particle indices and possibly other particle information.
 * @param a_internalPoPs            [in]    The *internal* PoPs::Database instance used to get particle indices and possibly other particle information.
 *                                          This is the <**PoPs**> node under the <**reactionSuite**> node.
 * @param a_name                    [in]    The moniker for the node to be parsed.
 * @param a_styles                  [in]    A pointer to the <**styles**> node.
 * @return                                  The parsed and constructed GIDI::Reaction instance.
 ***********************************************************************************************************/

Form *parseReaction( Construction::Settings const &a_construction, Suite *a_parent, pugi::xml_node const &a_node, PoPs::Database const &a_pops, 
                PoPs::Database const &a_internalPoPs, std::string const &a_name, Styles::Suite const *a_styles ) {

    return( parseReactionType( reactionMoniker, a_construction, a_parent, a_node, a_pops, a_internalPoPs, a_name, a_styles ) );
}

/* *********************************************************************************************************//**
 * Function that parses an <**orphanProduct**> node. Called from a Suite::parse instance.
 *
 * @param a_construction            [in]    Used to pass user options for parsing.
 * @param a_parent                  [in]    The parent GIDI::Suite that the returned Form will be added to.
 * @param a_node                    [in]    The **pugi::xml_node** to be parsed.
 * @param a_pops                    [in]    A PoPs Database instance used to get particle indices and possibly other particle information.
 * @param a_internalPoPs            [in]    The *internal* PoPs::Database instance used to get particle indices and possibly other particle information.
 *                                          This is the <**PoPs**> node under the <**reactionSuite**> node.
 * @param a_name                    [in]    The moniker for the node to be parsed.
 * @param a_styles                  [in]    A pointer to the <**styles**> node.
 * @return                                  The parsed and constructed GIDI::Reaction instance.
 ***********************************************************************************************************/

Form *parseOrphanProduct( Construction::Settings const &a_construction, Suite *a_parent, pugi::xml_node const &a_node, PoPs::Database const &a_pops,
                PoPs::Database const &a_internalPoPs, std::string const &a_name, Styles::Suite const *a_styles ) {

    return( parseReactionType( orphanProductMoniker, a_construction, a_parent, a_node, a_pops, a_internalPoPs, a_name, a_styles ) );
}

/* *********************************************************************************************************//**
 * Function that parses an <**orphanProduct**> node. Called from a Suite::parse instance.
 *
 * @param a_construction            [in]    Used to pass user options for parsing.
 * @param a_parent                  [in]    The parent GIDI::Suite that the returned Form will be added to.
 * @param a_node                    [in]    The **pugi::xml_node** to be parsed.
 * @param a_pops                    [in]    A PoPs Database instance used to get particle indices and possibly other particle information.
 * @param a_internalPoPs            [in]    The *internal* PoPs::Database instance used to get particle indices and possibly other particle information.
 *                                          This is the <**PoPs**> node under the <**reactionSuite**> node.
 * @param a_name                    [in]    The moniker for the node to be parsed.
 * @param a_styles                  [in]    A pointer to the <**styles**> node.
 * @return                                  The parsed and constructed GIDI::Reaction instance.
 ***********************************************************************************************************/

Form *parseFissionComponent( Construction::Settings const &a_construction, Suite *a_parent, pugi::xml_node const &a_node, PoPs::Database const &a_pops,
                PoPs::Database const &a_internalPoPs, std::string const &a_name, Styles::Suite const *a_styles ) {

    return( parseReactionType( fissionComponentMoniker, a_construction, a_parent, a_node, a_pops, a_internalPoPs, a_name, a_styles ) );
}

/* *********************************************************************************************************//**
 * Function that parses a <**reaction**> or an <**orphanProduct**> node. Called from a Suite::parse instance.
 *
 * @param a_moniker                 [in]    The moniker for the form to parse.
 * @param a_construction            [in]    Used to pass user options for parsing.
 * @param a_parent                  [in]    The parent GIDI::Suite that the returned Form will be added to.
 * @param a_node                    [in]    The **pugi::xml_node** to be parsed.
 * @param a_pops                    [in]    A PoPs Database instance used to get particle indices and possibly other particle information.
 * @param a_internalPoPs            [in]    The *internal* PoPs::Database instance used to get particle indices and possibly other particle information.
 *                                          This is the <**PoPs**> node under the <**reactionSuite**> node.
 * @param a_name                    [in]    The moniker for the node to be parsed.
 * @param a_styles                  [in]    A pointer to the <**styles**> node.
 * @return                                  The parsed and constructed GIDI::Reaction instance.
 ***********************************************************************************************************/

Form *parseReactionType( std::string const &a_moniker, Construction::Settings const &a_construction, Suite *a_parent, pugi::xml_node const &a_node, PoPs::Database const &a_pops,
                PoPs::Database const &a_internalPoPs, std::string const &a_name, Styles::Suite const *a_styles ) {

    Form *form = NULL;

    if( a_name == a_moniker ) {
        Protare const &protare( *dynamic_cast<Protare const *>( a_parent->root( ) ) );
        form = new Reaction( a_construction, a_node, a_pops, a_internalPoPs, protare, a_styles ); }
    else {                                  // This should never happend.
        std::cout << "parseReactionType: Ignoring '" << a_moniker << "' unsupported form '" << a_name << "'." << std::endl;
    }

    return( form );
}

/* *********************************************************************************************************//**
 * Function that parses a <**crossSectionSum**> node. Called from a Suite::parse instance.
 *
 * @param a_construction            [in]    Used to pass user options for parsing.
 * @param a_parent                  [in]    The parent GIDI::Suite that the returned Form will be added to.
 * @param a_node                    [in]    The **pugi::xml_node** to be parsed.
 * @param a_pops                    [in]    A PoPs Database instance used to get particle indices and possibly other particle information.
 * @param a_internalPoPs            [in]    The *internal* PoPs::Database instance used to get particle indices and possibly other particle information.
 *                                          This is the <**PoPs**> node under the <**reactionSuite**> node.
 * @param a_name                    [in]    The moniker for the node to be parsed.
 * @param a_styles                  [in]    A pointer to the <**styles**> node.
 * @return                                  The parsed and constructed GIDI::CrossSectionSum instance.
 ***********************************************************************************************************/

Form *parseSumsCrossSectionsSuite( Construction::Settings const &a_construction, Suite *a_parent, pugi::xml_node const &a_node, PoPs::Database const &a_pops, 
                PoPs::Database const &a_internalPoPs, std::string const &a_name, Styles::Suite const *a_styles ) {

    Form *form = NULL;

    if( a_name == crossSectionSumMoniker ) {
        form = new Sums::CrossSectionSum( a_construction, a_node, a_pops, a_internalPoPs ); }
    else {                                  // This should never happend.
        std::cout << "parseSumsCrossSectionsSuite: Ignoring unsupported Form '" << a_name << "'." << std::endl;
    }

    return( form );
}

/* *********************************************************************************************************//**
 * Function that parses a <**multiplicitySum**> node. Called from a Suite::parse instance.
 *
 * @param a_construction            [in]    Used to pass user options for parsing.
 * @param a_parent                  [in]    The parent GIDI::Suite that the returned Form will be added to.
 * @param a_node                    [in]    The **pugi::xml_node** to be parsed.
 * @param a_pops                    [in]    A PoPs Database instance used to get particle indices and possibly other particle information.
 * @param a_internalPoPs            [in]    The *internal* PoPs::Database instance used to get particle indices and possibly other particle information.
 *                                          This is the <**PoPs**> node under the <**reactionSuite**> node.
 * @param a_name                    [in]    The moniker for the node to be parsed.
 * @param a_styles                  [in]    A pointer to the <**styles**> node.
 * @return                                  The parsed and constructed GIDI::MultiplicitySum instance.
 ***********************************************************************************************************/

Form *parseSumsMultiplicitiesSuite( Construction::Settings const &a_construction, Suite *a_parent, pugi::xml_node const &a_node, PoPs::Database const &a_pops, 
                PoPs::Database const &a_internalPoPs, std::string const &a_name, Styles::Suite const *a_styles ) {

    Form *form = NULL;

    if( a_construction.parseMode( ) == Construction::e_outline ) return( NULL );

    if( a_name == multiplicitySumMoniker ) {
        form = new Sums::MultiplicitySum( a_construction, a_node, a_pops, a_internalPoPs ); }
    else {                                  // This should never happend.
        std::cout << "parseSumsMultiplicitiesSuite: Ignoring unsupported Form '" << a_name << "'." << std::endl;
    }

    return( form );
}

/* *********************************************************************************************************//**
 * Function that parses a node under the <**doubleDifferentialCrossSection**> node. Called from a Suite::parse instance.
 *
 * @param a_construction            [in]    Used to pass user options for parsing.
 * @param a_parent                  [in]    The parent GIDI::Suite that the returned Form will be added to.
 * @param a_node                    [in]    The **pugi::xml_node** to be parsed.
 * @param a_pops                    [in]    A PoPs Database instance used to get particle indices and possibly other particle information.
 * @param a_internalPoPs            [in]    The *internal* PoPs::Database instance used to get particle indices and possibly other particle information.
 *                                          This is the <**PoPs**> node under the <**reactionSuite**> node.
 * @param a_name                    [in]    The moniker for the node to be parsed.
 * @param a_styles                  [in]    A pointer to the <**styles**> node.
 * @return                                  The parsed and constructed GIDI::Form or NULL if the node is not supported.
 ***********************************************************************************************************/

Form *parseDoubleDifferentialCrossSectionSuite( Construction::Settings const &a_construction, Suite *a_parent, pugi::xml_node const &a_node, PoPs::Database const &a_pops, 
                PoPs::Database const &a_internalPoPs, std::string const &a_name, Styles::Suite const *a_styles ) {

    if( a_construction.parseMode( ) == Construction::e_outline ) return( NULL );
    if( a_construction.parseMode( ) == Construction::e_multiGroupOnly ) return( NULL );

    Form *form = NULL;

    if( a_name == coherentPhotonScatteringMoniker ) {
        form = new DoubleDifferentialCrossSection::CoherentPhotoAtomicScattering( a_construction, a_node, a_pops, a_internalPoPs, a_parent ); }
    else if( a_name == incoherentPhotonScatteringMoniker ) {
        form = new DoubleDifferentialCrossSection::IncoherentPhotoAtomicScattering( a_construction, a_node, a_pops, a_internalPoPs, a_parent ); }
    else if( a_name == TNSL_coherentElasticMoniker ) {
        form = new DoubleDifferentialCrossSection::n_ThermalNeutronScatteringLaw::CoherentElastic( a_construction, a_node, a_pops, a_internalPoPs, a_parent ); }
    else if( a_name == TNSL_incoherentElasticMoniker ) {
        form = new DoubleDifferentialCrossSection::n_ThermalNeutronScatteringLaw::IncoherentElastic( a_construction, a_node, a_pops, a_internalPoPs, a_parent ); }
    else if( a_name == TNSL_incoherentInelasticMoniker ) {
        form = new DoubleDifferentialCrossSection::n_ThermalNeutronScatteringLaw::IncoherentInelastic( a_construction, a_node, a_pops, a_internalPoPs, a_parent ); }
    else if( a_name == CoulombPlusNuclearElasticMoniker ) { 
        }
    else {
        std::cout << "parseDoubleDifferentialCrossSectionSuite: Ignoring unsupported Form '" << a_name << "'." << std::endl;
    }

    return( form );
}

/* *********************************************************************************************************//**
 * Function that parses a node under the <**doubleDifferentialCrossSection**> node. Called from a Suite::parse instance.
 *
 * @param a_construction            [in]    Used to pass user options for parsing.
 * @param a_parent                  [in]    The parent GIDI::Suite that the returned Form will be added to.
 * @param a_node                    [in]    The **pugi::xml_node** to be parsed.
 * @param a_pops                    [in]    A PoPs Database instance used to get particle indices and possibly other particle information.
 * @param a_internalPoPs            [in]    The *internal* PoPs::Database instance used to get particle indices and possibly other particle information.
 *                                          This is the <**PoPs**> node under the <**reactionSuite**> node.
 * @param a_name                    [in]    The moniker for the node to be parsed.
 * @param a_styles                  [in]    A pointer to the <**styles**> node.
 * @return                                  The parsed and constructed GIDI::Form or NULL if the node is not supported.
 ***********************************************************************************************************/

Form *parseScatteringAtom( Construction::Settings const &a_construction, Suite *a_parent, pugi::xml_node const &a_node, PoPs::Database const &a_pops,
                PoPs::Database const &a_internalPoPs, std::string const &a_name, Styles::Suite const *a_styles ) {

    return( new DoubleDifferentialCrossSection::n_ThermalNeutronScatteringLaw::ScatteringAtom( a_construction, a_node ) );
}

/* *********************************************************************************************************//**
 * Function that parses a node under the <**crossSection**> node. Called from a Suite::parse instance.
 *
 * @param a_construction            [in]    Used to pass user options for parsing.
 * @param a_parent                  [in]    The parent GIDI::Suite that the returned Form will be added to.
 * @param a_node                    [in]    The **pugi::xml_node** to be parsed.
 * @param a_pops                    [in]    A PoPs Database instance used to get particle indices and possibly other particle information.
 * @param a_internalPoPs            [in]    The *internal* PoPs::Database instance used to get particle indices and possibly other particle information.
 *                                          This is the <**PoPs**> node under the <**reactionSuite**> node.
 * @param a_name                    [in]    The moniker for the node to be parsed.
 * @param a_styles                  [in]    A pointer to the <**styles**> node.
 * @return                                  The parsed and constructed GIDI::Form or NULL if the node is not supported.
 ***********************************************************************************************************/

Form *parseCrossSectionSuite( Construction::Settings const &a_construction, Suite *a_parent, pugi::xml_node const &a_node, PoPs::Database const &a_pops, 
                PoPs::Database const &a_internalPoPs, std::string const &a_name, Styles::Suite const *a_styles ) {

    if( a_construction.parseMode( ) == Construction::e_outline ) return( NULL );
    if( ( a_construction.parseMode( ) == Construction::e_multiGroupOnly ) && ( a_name != gridded1dMoniker ) ) return( NULL );
    if( ( a_construction.parseMode( ) == Construction::e_MonteCarloContinuousEnergy ) && ( a_name != Ys1dMoniker ) ) return( NULL );

// Form not parsed is CoulombPlusNuclearElastic.
    Form *form = NULL;

    if( a_name == resonancesWithBackgroundMoniker ) {
        return( new ResonancesWithBackground1d( a_construction, a_node, a_parent ) ); }
    else if( a_name == TNSL1dMoniker ) {
        form = new ThermalNeutronScatteringLaw1d( a_construction, a_node, a_parent ); }
    else if( a_name == URR_probabilityTables1ddMoniker ) {
        form = new URR_probabilityTables1d( a_construction, a_node, a_parent ); }
    else if( a_name == CoulombPlusNuclearElasticMoniker ) {
        }
    else if( a_name == CoulombPlusNuclearElasticMoniker ) {
        }
    else {
        form = data1dParse( a_construction, a_node, a_parent );
    }

    return( form );
}

/* *********************************************************************************************************//**
 * Function that parses a node under the <**DelayedNeutrons**> node. Called from a Suite::parse instance.
 *
 * @param a_construction            [in]    Used to pass user options for parsing.
 * @param a_parent                  [in]    The parent GIDI::Suite that the returned Form will be added to.
 * @param a_node                    [in]    The **pugi::xml_node** to be parsed.
 * @param a_pops                    [in]    A PoPs Database instance used to get particle indices and possibly other particle information.
 * @param a_internalPoPs            [in]    The *internal* PoPs::Database instance used to get particle indices and possibly other particle information.
 *                                          This is the <**PoPs**> node under the <**reactionSuite**> node.
 * @param a_name                    [in]    The moniker for the node to be parsed.
 * @param a_styles                  [in]    A pointer to the <**styles**> node.
 * @return                                  The parsed and constructed GIDI::Form or NULL if the node is not supported.
 ***********************************************************************************************************/

Form *parseDelayedNeutronsSuite( Construction::Settings const &a_construction, Suite *a_parent, pugi::xml_node const &a_node, PoPs::Database const &a_pops,
                PoPs::Database const &a_internalPoPs, std::string const &a_name, Styles::Suite const *a_styles ) {

    return( new DelayedNeutron( a_construction, a_node, a_pops, a_internalPoPs, a_parent, a_styles ) );
}

/* *********************************************************************************************************//**
 * Function that parses a node under the <**FissionEnergyReleases**> node. Called from a Suite::parse instance.
 *
 * @param a_construction            [in]    Used to pass user options for parsing.
 * @param a_parent                  [in]    The parent GIDI::Suite that the returned Form will be added to.
 * @param a_node                    [in]    The **pugi::xml_node** to be parsed.
 * @param a_pops                    [in]    A PoPs Database instance used to get particle indices and possibly other particle information.
 * @param a_internalPoPs            [in]    The *internal* PoPs::Database instance used to get particle indices and possibly other particle information.
 *                                          This is the <**PoPs**> node under the <**reactionSuite**> node.
 * @param a_name                    [in]    The moniker for the node to be parsed.
 * @param a_styles                  [in]    A pointer to the <**styles**> node.
 * @return                                  The parsed and constructed GIDI::Form or NULL if the node is not supported.
 ***********************************************************************************************************/

Form *parseFissionEnergyReleasesSuite( Construction::Settings const &a_construction, Suite *a_parent, pugi::xml_node const &a_node, PoPs::Database const &a_pops,
                PoPs::Database const &a_internalPoPs, std::string const &a_name, Styles::Suite const *a_styles ) {

    return( new FissionEnergyRelease( a_construction, a_node, a_parent ) );
}

/* *********************************************************************************************************//**
 * Function that parses a node under the <**rate**> node. Called from a Suite::parse instance.
 *
 * @param a_construction            [in]    Used to pass user options for parsing.
 * @param a_parent                  [in]    The parent GIDI::Suite that the returned Form will be added to.
 * @param a_node                    [in]    The **pugi::xml_node** to be parsed.
 * @param a_pops                    [in]    A PoPs Database instance used to get particle indices and possibly other particle information.
 * @param a_internalPoPs            [in]    The *internal* PoPs::Database instance used to get particle indices and possibly other particle information.
 *                                          This is the <**PoPs**> node under the <**reactionSuite**> node.
 * @param a_name                    [in]    The moniker for the node to be parsed.
 * @param a_styles                  [in]    A pointer to the <**styles**> node.
 * @return                                  The parsed and constructed GIDI::Form or NULL if the node is not supported.
 ***********************************************************************************************************/

Form *parseRateSuite( Construction::Settings const &a_construction, Suite *a_parent, pugi::xml_node const &a_node, PoPs::Database const &a_pops,
                PoPs::Database const &a_internalPoPs, std::string const &a_name, Styles::Suite const *a_styles ) {

    return( new Rate( a_construction, a_node, a_parent ) );
}

/* *********************************************************************************************************//**
 * Function that parses a node under an <**availableEnergy**> or <**availableMomentum**> node. Called from a Suite::parse instance.
 *
 * @param a_construction            [in]    Used to pass user options for parsing.
 * @param a_parent                  [in]    The parent GIDI::Suite that the returned Form will be added to.
 * @param a_node                    [in]    The **pugi::xml_node** to be parsed.
 * @param a_pops                    [in]    A PoPs Database instance used to get particle indices and possibly other particle information.
 * @param a_internalPoPs            [in]    The *internal* PoPs::Database instance used to get particle indices and possibly other particle information.
 *                                          This is the <**PoPs**> node under the <**reactionSuite**> node.
 * @param a_name                    [in]    The moniker for the node to be parsed.
 * @param a_styles                  [in]    A pointer to the <**styles**> node.
 * @return                                  The parsed and constructed GIDI::Function1d instance.
 ***********************************************************************************************************/

Form *parseAvailableSuite( Construction::Settings const &a_construction, Suite *a_parent, pugi::xml_node const &a_node, PoPs::Database const &a_pops, 
                PoPs::Database const &a_internalPoPs, std::string const &a_name, Styles::Suite const *a_styles ) {

    if( a_construction.parseMode( ) == Construction::e_outline ) return( NULL );
    if( ( a_construction.parseMode( ) == Construction::e_multiGroupOnly ) && ( a_name != gridded1dMoniker ) ) return( NULL );

    return( data1dParse( a_construction, a_node, a_parent ) );
}

/* *********************************************************************************************************//**
 * Function that parses a node under a <**Q**> node. Called from a Suite::parse instance.
 *
 * @param a_construction            [in]    Used to pass user options for parsing.
 * @param a_parent                  [in]    The parent GIDI::Suite that the returned Form will be added to.
 * @param a_node                    [in]    The **pugi::xml_node** to be parsed.
 * @param a_pops                    [in]    A PoPs Database instance used to get particle indices and possibly other particle information.
 * @param a_internalPoPs            [in]    The *internal* PoPs::Database instance used to get particle indices and possibly other particle information.
 *                                          This is the <**PoPs**> node under the <**reactionSuite**> node.
 * @param a_name                    [in]    The moniker for the node to be parsed.
 * @param a_styles                  [in]    A pointer to the <**styles**> node.
 * @return                                  The parsed and constructed GIDI::Function1d instance.
 ***********************************************************************************************************/

Form *parseQSuite( Construction::Settings const &a_construction, Suite *a_parent, pugi::xml_node const &a_node, PoPs::Database const &a_pops, 
                PoPs::Database const &a_internalPoPs, std::string const &a_name, Styles::Suite const *a_styles ) {

    Form *form = NULL;

    if( a_construction.parseMode( ) == Construction::e_outline ) return( NULL );

    form = data1dParse( a_construction, a_node, a_parent );

    return( form );
}

/* *********************************************************************************************************//**
 * Function that parses a node under a <**products**> node. Called from a Suite::parse instance.
 *
 * @param a_construction            [in]    Used to pass user options for parsing.
 * @param a_parent                  [in]    The parent GIDI::Suite that the returned Form will be added to.
 * @param a_node                    [in]    The **pugi::xml_node** to be parsed.
 * @param a_pops                    [in]    A PoPs Database instance used to get particle indices and possibly other particle information.
 * @param a_internalPoPs            [in]    The *internal* PoPs::Database instance used to get particle indices and possibly other particle information.
 *                                          This is the <**PoPs**> node under the <**reactionSuite**> node.
 * @param a_name                    [in]    The moniker for the node to be parsed.
 * @param a_styles                  [in]    A pointer to the <**styles**> node.
 * @return                                  The parsed and constructed GIDI::Product instance.
 ***********************************************************************************************************/

Form *parseProductSuite( Construction::Settings const &a_construction, Suite *a_parent, pugi::xml_node const &a_node, PoPs::Database const &a_pops, 
                PoPs::Database const &a_internalPoPs, std::string const &a_name, Styles::Suite const *a_styles ) {

    Form *form = NULL;

    if( a_name == productMoniker ) {
        form = new Product( a_construction, a_node, a_pops, a_internalPoPs, a_parent, a_styles ); }
    else {
        std::cout << "parseProductSuite: Ignoring unsupported element in products " << a_node.name( ) << std::endl;
    }

    return( form );
}

/* *********************************************************************************************************//**
 * Function that parses a node under a <**multiplicity**> node. Called from a Suite::parse instance.
 *
 * @param a_construction            [in]    Used to pass user options for parsing.
 * @param a_parent                  [in]    The parent GIDI::Suite that the returned Form will be added to.
 * @param a_node                    [in]    The **pugi::xml_node** to be parsed.
 * @param a_pops                    [in]    A PoPs Database instance used to get particle indices and possibly other particle information.
 * @param a_internalPoPs            [in]    The *internal* PoPs::Database instance used to get particle indices and possibly other particle information.
 *                                          This is the <**PoPs**> node under the <**reactionSuite**> node.
 * @param a_name                    [in]    The moniker for the node to be parsed.
 * @param a_styles                  [in]    A pointer to the <**styles**> node.
 * @return                                  The parsed and constructed GIDI::Function1d instance.
 ***********************************************************************************************************/

Form *parseMultiplicitySuite( Construction::Settings const &a_construction, Suite *a_parent, pugi::xml_node const &a_node, PoPs::Database const &a_pops, 
                PoPs::Database const &a_internalPoPs, std::string const &a_name, Styles::Suite const *a_styles ) {

    if( a_construction.parseMode( ) == Construction::e_outline ) return( NULL );

    if( a_name == branching1dMoniker ) return( new Branching1d( a_construction, a_node, a_parent ) );

    return( data1dParse( a_construction, a_node, a_parent ) );
}

/* *********************************************************************************************************//**
 * Function that parses a node under a <**distribution**> node. Called from a Suite::parse instance.
 *
 * @param a_construction            [in]    Used to pass user options for parsing.
 * @param a_parent                  [in]    The parent GIDI::Suite that the returned Form will be added to.
 * @param a_node                    [in]    The **pugi::xml_node** to be parsed.
 * @param a_pops                    [in]    A PoPs Database instance used to get particle indices and possibly other particle information.
 * @param a_internalPoPs            [in]    The *internal* PoPs::Database instance used to get particle indices and possibly other particle information.
 *                                          This is the <**PoPs**> node under the <**reactionSuite**> node.
 * @param a_name                    [in]    The moniker for the node to be parsed.
 * @param a_styles                  [in]    A pointer to the <**styles**> node.
 * @return                                  The parsed and constructed GIDI::Form or NULL if the node is not supported.
 ***********************************************************************************************************/

Form *parseDistributionSuite( Construction::Settings const &a_construction, Suite *a_parent, pugi::xml_node const &a_node, PoPs::Database const &a_pops, 
                PoPs::Database const &a_internalPoPs, std::string const &a_name, Styles::Suite const *a_styles ) {

    if( a_construction.parseMode( ) == Construction::e_outline ) return( NULL );
    if( a_name == multiGroup3dMoniker ) {
        if( ( a_construction.parseMode( ) == Construction::e_MonteCarloContinuousEnergy ) || 
            ( a_construction.parseMode( ) == Construction::e_excludeProductMatrices ) ) return( NULL ); }
    else {
        if( a_construction.parseMode( ) == Construction::e_multiGroupOnly ) return( NULL );
    }

//  Distributions not parsed are CoulombPlusNuclearElasticMoniker and LLNLLegendreMoniker.
    Form *form = NULL;

    if( a_name == multiGroup3dMoniker ) {
        form = new MultiGroup3d( a_construction, a_node, a_parent ); }
    else if( a_name == angularTwoBodyMoniker ) {
        form = new AngularTwoBody( a_construction, a_node, a_parent ); }
    else if( a_name == uncorrelatedMoniker ) {
        form = new Uncorrelated( a_construction, a_node, a_parent ); }
    else if( a_name == KalbachMannMoniker ) {
        form = new KalbachMann( a_construction, a_node, a_parent ); }
    else if( a_name == energyAngularMoniker ) {
        form = new EnergyAngular( a_construction, a_node, a_parent ); }
    else if( a_name == energyAngularMCMoniker ) {
        form = new EnergyAngularMC( a_construction, a_node, a_parent ); }
    else if( a_name == angularEnergyMoniker ) {
        form = new AngularEnergy( a_construction, a_node, a_parent ); }
    else if( a_name == angularEnergyMCMoniker ) {
        form = new AngularEnergyMC( a_construction, a_node, a_parent ); }
    else if( a_name == LLNLAngularEnergyMoniker ) {
        form = new LLNLAngularEnergy( a_construction, a_node, a_parent ); }
    else if( a_name == coherentPhotonScatteringMoniker ) {
        form = new CoherentPhotoAtomicScattering( a_construction, a_node, a_parent ); }
    else if( a_name == incoherentPhotonScatteringMoniker ) {
        form = new IncoherentPhotoAtomicScattering( a_construction, a_node, a_parent ); }
    else if( a_name == TNSLMoniker ) {
        form = new ThermalNeutronScatteringLaw( a_construction, a_node, a_parent ); }
    else if( a_name == branching3dMoniker ) {
        form = new Branching3d( a_construction, a_node, a_parent ); }
    else if( a_name == referenceMoniker ) {
        form = new Reference3d( a_construction, a_node, a_parent ); }
    else if( a_name == unspecifiedMoniker ) {
        form = new Unspecified( a_construction, a_node, a_parent ); }
    else if( a_name == CoulombPlusNuclearElasticMoniker ) {
        }
    else if( a_name == LLNLLegendreMoniker ) {
        }
    else {
        std::cout << "parseDistributionSuite: Ignoring unsupported distribution " << a_node.name( ) << std::endl;
    }

    return( form );
}

/* *********************************************************************************************************//**
 * Function that parses a node under an <**averageEnergy**> node. Called from a Suite::parse instance.
 *
 * @param a_construction            [in]    Used to pass user options for parsing.
 * @param a_parent                  [in]    The parent GIDI::Suite that the returned Form will be added to.
 * @param a_node                    [in]    The **pugi::xml_node** to be parsed.
 * @param a_pops                    [in]    A PoPs Database instance used to get particle indices and possibly other particle information.
 * @param a_internalPoPs            [in]    The *internal* PoPs::Database instance used to get particle indices and possibly other particle information.
 *                                          This is the <**PoPs**> node under the <**reactionSuite**> node.
 * @param a_name                    [in]    The moniker for the node to be parsed.
 * @param a_styles                  [in]    A pointer to the <**styles**> node.
 * @return                                  The parsed and constructed GIDI::Function1d instance.
 ***********************************************************************************************************/

Form *parseAverageEnergySuite( Construction::Settings const &a_construction, Suite *a_parent, pugi::xml_node const &a_node, PoPs::Database const &a_pops, 
                PoPs::Database const &a_internalPoPs, std::string const &a_name, Styles::Suite const *a_styles ) {

    if( a_construction.parseMode( ) == Construction::e_outline ) return( NULL );
    if( ( a_construction.parseMode( ) == Construction::e_multiGroupOnly ) && ( a_name != gridded1dMoniker ) ) return( NULL );

    return( data1dParse( a_construction, a_node, a_parent ) );
}

/* *********************************************************************************************************//**
 * Function that parses a node under an <**averageMomentum**> node. Called from a Suite::parse instance.
 *
 * @param a_construction            [in]    Used to pass user options for parsing.
 * @param a_parent                  [in]    The parent GIDI::Suite that the returned Form will be added to.
 * @param a_node                    [in]    The **pugi::xml_node** to be parsed.
 * @param a_pops                    [in]    A PoPs Database instance used to get particle indices and possibly other particle information.
 * @param a_internalPoPs            [in]    The *internal* PoPs::Database instance used to get particle indices and possibly other particle information.
 *                                          This is the <**PoPs**> node under the <**reactionSuite**> node.
 * @param a_name                    [in]    The moniker for the node to be parsed.
 * @param a_styles                  [in]    A pointer to the <**styles**> node.
 *
 * @return                                  The parsed and constructed GIDI::Function1d instance.
 ***********************************************************************************************************/

Form *parseAverageMomentumSuite( Construction::Settings const &a_construction, Suite *a_parent, pugi::xml_node const &a_node, PoPs::Database const &a_pops, 
                PoPs::Database const &a_internalPoPs, std::string const &a_name, Styles::Suite const *a_styles ) {

    if( a_construction.parseMode( ) == Construction::e_outline ) return( NULL );
    if( ( a_construction.parseMode( ) == Construction::e_multiGroupOnly ) && ( a_name != gridded1dMoniker ) ) return( NULL );

    return( data1dParse( a_construction, a_node, a_parent ) );
}

/* *********************************************************************************************************//**
 * Function that parses a node one-d function node. Called from a Suite::parse instance.
 *
 * @param a_construction            [in]    Used to pass user options for parsing.
 * @param a_node            [in]    The **pugi::xml_node** to be parsed.
 * @param a_parent          [in]    The parent GIDI::Suite that the returned Form will be added to.
 *
 * @return                          The parsed and constructed GIDI::Function1d instance.
 ***********************************************************************************************************/

Function1dForm *data1dParse( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent ) {

    Function1dForm *form = NULL;
    std::string name( a_node.name( ) );

    if( name == constant1dMoniker ) {
        form = new Constant1d( a_construction, a_node, a_parent ); }
    else if( name == XYs1dMoniker ) {
        form = new XYs1d( a_construction, a_node, a_parent ); }
    else if( name == Ys1dMoniker ) {
        form = new Ys1d( a_construction, a_node, a_parent ); }
    else if( name == polynomial1dMoniker ) {
        form = new Polynomial1d( a_construction, a_node, a_parent ); }
    else if( name == LegendreMoniker ) {
        form = new Legendre1d( a_construction, a_node, a_parent ); }
    else if( name == regions1dMoniker ) {
        form = new Regions1d( a_construction, a_node, a_parent ); }
    else if( name == gridded1dMoniker ) {
        form = new Gridded1d( a_construction, a_node, a_parent ); }
    else if( name == referenceMoniker ) {
        form = new Reference1d( a_construction, a_node, a_parent ); }
    else if( name == xs_pdf_cdf1dMoniker ) {
        form = new Xs_pdf_cdf1d( a_construction, a_node, a_parent ); }
    else if( name == unspecified1dMoniker ) {
        form = new Unspecified1d( a_construction, a_node, a_parent ); }
    else {
        std::cout << "data1dParse: Ignoring unsupported 1d function = '" << name << "'" << std::endl;
    }

    return( form );
}

/* *********************************************************************************************************//**
 * Function that parses a node one-d function node. Called from a Suite::parse instance. If no node exists, returns NULL.
 *
 * @param a_construction    [in]    Used to pass user options to the constructor.
 * @param a_node            [in]    The **pugi::xml_node** to be parsed.
 * @param a_parent          [in]    The parent GIDI::Suite that the returned Form will be added to.
 *
 * @return                          The parsed and constructed GIDI::Function1d instance.
 ***********************************************************************************************************/

Function1dForm *data1dParseAllowEmpty( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent ) {

    std::string name( a_node.name( ) );

    if( name == "" ) return( NULL );
    return( data1dParse( a_construction, a_node, a_parent ) );
}


/* *********************************************************************************************************//**
 * Function that parses a node two-d function node. Called from a Suite::parse instance.
 *
 * @param a_construction    [in]    Used to pass user options to the constructor.
 * @param a_node            [in]    The **pugi::xml_node** to be parsed.
 * @param a_parent          [in]    The parent GIDI::Suite that the returned Form will be added to.
 * @return                          The parsed and constructed GIDI::Function2d instance.
 ***********************************************************************************************************/
 
Function2dForm *data2dParse( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent ) {

    Function2dForm *form = NULL;
    std::string name( a_node.name( ) );

    if( name == XYs2dMoniker ) {
        form = new XYs2d( a_construction, a_node, a_parent ); }
    else if( name == recoilMoniker ) {
        form = new Recoil2d( a_construction, a_node, a_parent ); }
    else if( name == isotropic2dMoniker ) {
        form = new Isotropic2d( a_construction, a_node, a_parent ); }
    else if( name == discreteGammaMoniker ) {
        form = new DiscreteGamma2d( a_construction, a_node, a_parent ); }
    else if( name == primaryGammaMoniker ) {
        form = new PrimaryGamma2d( a_construction, a_node, a_parent ); }
    else if( name == generalEvaporationMoniker ) {
        form = new GeneralEvaporation2d( a_construction, a_node, a_parent ); }
    else if( name == simpleMaxwellianFissionMoniker ) {
        form = new SimpleMaxwellianFission2d( a_construction, a_node, a_parent ); }
    else if( name == evaporationMoniker ) {
        form = new Evaporation2d( a_construction, a_node, a_parent ); }
    else if( name == WattMoniker ) {
        form = new Watt2d( a_construction, a_node, a_parent ); }
    else if( name == MadlandNixMoniker ) {
        form = new MadlandNix2d( a_construction, a_node, a_parent ); }
    else if( name == weightedFunctionalsMoniker ) {
        form = new WeightedFunctionals2d( a_construction, a_node, a_parent ); }
    else if( name == NBodyPhaseSpaceMoniker ) {
        form = new NBodyPhaseSpace2d( a_construction, a_node, a_parent ); }
    else if( name == regions2dMoniker ) {
        form = new Regions2d( a_construction, a_node, a_parent ); }
    else {
        std::cout << "data2dParse: Ignoring unsupported 2d function = '" << name << "'" << std::endl;
    }

    return( form );
}

/* *********************************************************************************************************//**
 * Function that parses a node three-d function node. Called from a Suite::parse instance.
 *
 * @param a_construction    [in]    Used to pass user options to the constructor.
 * @param a_node            [in]    The **pugi::xml_node** to be parsed.
 * @param a_parent          [in]    The parent GIDI::Suite that the returned Form will be added to.
 * @return                          The parsed and constructed GIDI::Function3d instance.
 ***********************************************************************************************************/
 
Function3dForm *data3dParse( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent ) {

    Function3dForm *form = NULL;
    std::string name( a_node.name( ) );

    if( name == XYs3dMoniker ) {
        form = new XYs3d( a_construction, a_node, a_parent ); }
    else if( name == gridded3dMoniker ) {
        form = new Gridded3d( a_construction, a_node ); }
    else {
        std::cout << "data3dParse: Ignoring unsupported 3d function = '" << name << "'" << std::endl;
    }
    
    return( form );
}

}
