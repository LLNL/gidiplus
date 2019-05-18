/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <set>

#include "GIDI.hpp"

void readProtare( GIDI::Map &map, PoPs::Database const &pops, std::string const &targetID );
void printFunctionInfo( std::string const &name, GIDI::Function1dForm const *function );
double integrateSub( int _n, double _a, double logE, double E1, double _y1, double E2, double _y2 );
std::string toString( double a_value, char const *a_fmt = NULL );
/*
=========================================================
*/
int main( int argc, char **argv ) {

    PoPs::Database pops( "../pops.xml" );
    std::string mapFilename( "../all.map" );
    GIDI::Map map( mapFilename, pops );

    if( argc > 1 ) mapFilename = "../Data/MG_MC/all_maps.map";

    std::cerr << "    " << __FILE__;
    for( int i1 = 1; i1 < argc; i1++ ) std::cerr << " " << argv[i1];
    std::cerr << std::endl;

    readProtare( map, pops, "O16" );
    readProtare( map, pops, "Th229" );
}
/*
=========================================================
*/
void readProtare( GIDI::Map &map, PoPs::Database const &pops, std::string const &targetID ) {

    GIDI::Protare *protare;

    try {
        GIDI::Construction::Settings construction( GIDI::Construction::e_all, GIDI::Construction::e_atomicOnly );
        protare = map.protare( construction, pops, PoPs::IDs::photon, targetID); }
    catch (char const *str) {
        std::cout << str << std::endl;
        exit( EXIT_FAILURE );
    }
    if( protare == NULL ) {
        std::cout << "protare for " << targetID << " not found." << std::endl;
        exit( EXIT_FAILURE );
    }

    std::string fileName( protare->fileName( ) );
    std::size_t offset = fileName.rfind( "GIDI" );
    std::cout << fileName.substr( offset ) << std::endl;

    for( std::size_t index = 0; index < protare->numberOfReactions( ); ++index ) {
        GIDI::Reaction const *reaction = protare->reaction( index );
        GIDI::OutputChannel const *outputChannel = reaction->outputChannel( );
        GIDI::Product const *product = static_cast<GIDI::Product const *>( outputChannel->products( ).get<GIDI::Product>( 0 ) );
        GIDI::Suite const &distribution = product->distribution( );

        std::cout << "    " << reaction->label( ) << std::endl;
        std::cout << "      " << product->particle( ).ID( ) << std::endl;

        for( std::size_t i2 = 0; i2 < distribution.size( ); ++i2 ) {
            GIDI::Distribution const *form = distribution.get<GIDI::Distribution>( i2 );

            std::cout << "        distribution form moniker = " << form->moniker( ) << std::endl;

            if( form->moniker( ) == coherentPhotonScatteringMoniker ) {
                GIDI::CoherentPhotoAtomicScattering const *coherentPhotonScattering = static_cast<GIDI::CoherentPhotoAtomicScattering const *>( form );

                std::cout << "          href = " << coherentPhotonScattering->href( ) << std::endl;

                GIDI::Ancestry const *link = coherentPhotonScattering->findInAncestry( coherentPhotonScattering->href( ) );

                GIDI::DoubleDifferentialCrossSection::CoherentPhotoAtomicScattering const *dd = static_cast<GIDI::DoubleDifferentialCrossSection::CoherentPhotoAtomicScattering const *>( link );
                std::cout << "          dd moniker = " << dd->moniker( ) << std::endl;

                printFunctionInfo( "formFactor", dd->formFactor( ) );
                printFunctionInfo( "realAnomalousFactor", dd->realAnomalousFactor( ) );
                printFunctionInfo( "imaginaryAnomalousFactor", dd->imaginaryAnomalousFactor( ) );
            }
        }
    }    

    delete protare;
}
/*
=========================================================
*/
void printFunctionInfo( std::string const &name, GIDI::Function1dForm const *function ) {

    char xFmt[] = "%23.16e";

    if( function == NULL ) return;

    std::string domainUnit = function->domainUnit( );
    std::string rangeUnit = function->rangeUnit( );
    double domainFactor = 1.0;

    if( domainUnit == "1/Ang" ) {
        domainFactor = 0.012398419739640716; }              // Converts 'h * c /Ang' to MeV.
    else if( domainUnit == "1/cm" ) {
        domainFactor = 0.012398419739640716 * 1e-8; }              // Converts 'h * c /cm' to MeV.
    else if( domainUnit == "MeV" ) {
        }
    else {
        std::cout << "WARNING: unsupported domain unit = '" << domainUnit << "'" << std::endl;
    }


    std::cout << "            name = " << name << std::endl;
    std::cout << "              domain unit = '" << domainUnit << "' range unit = '" << rangeUnit << "'" << std::endl;

    if( name == "formFactor" ) {
        if( function->type( ) != GIDI::f_regions1d ) {
            std::cout << "WARNING: unsupported data type = " << function->type( ) << std::endl;
            return;
        }

        GIDI::Regions1d *regions1d = (GIDI::Regions1d *)( function );
        for( std::size_t i1 = 0; i1 < regions1d->Xs( ).size( ); ++i1 ) std::cout << "              x[" << i1 << "] = " << regions1d->Xs( )[i1] << std::endl;

        std::vector<GIDI::Function1dForm *> &functions1d = regions1d->functions1d2( );
        std::cout << "              number of regions = " << functions1d.size( );
        for( std::size_t i1 = 0; i1 < functions1d.size( ); ++i1 ) {
            GIDI::XYs1d const &xys1d = *static_cast<GIDI::XYs1d const *>( functions1d[i1] );

            std::cout << "              interpolation = " << xys1d.interpolationString( ) << std::endl;
        }
        if( functions1d.size( ) == 2 ) {
            GIDI::XYs1d const &xys1d = *static_cast<GIDI::XYs1d const *>( functions1d[0] );
            std::vector<double> xs = xys1d.xs( );
            std::vector<double> ys = xys1d.ys( );
            double sum1 = 0.0, sum2 = 0.0;

            std::vector<double> Es( xs );
            for( std::size_t i1 = 0; i1 < Es.size( ); ++i1 ) Es[i1] *= domainFactor;

            std::cout << "                :: " << toString( xs[0], xFmt ) << "  " << toString( Es[0], xFmt ) << "  " << toString( ys[0] ) << "  " << toString( sum1 ) << "  " << toString( sum2 ) << std::endl;
            sum1 += 0.5 * Es[1] * Es[1] * ys[1];
            sum2 += 0.5 * Es[1] * Es[1] * ys[1] * ys[1];
            std::cout << "                :: " << toString( xs[1], xFmt ) << "  " << toString( Es[1], xFmt ) << "  " << toString( ys[1] ) << "  " << toString( sum1 ) << "  " << toString( sum2 ) << std::endl;

            GIDI::XYs1d const &xys1d2 = *static_cast<GIDI::XYs1d const *>( functions1d[1] );
            xs = xys1d2.xs( );
            Es = xs;
            for( std::size_t i1 = 0; i1 < Es.size( ); ++i1 ) Es[i1] *= domainFactor;
            ys = xys1d2.ys( );

            double E1 = Es[0], y1 = ys[0];
            for( std::size_t i1 = 1; i1 < xs.size( ); ++i1 ) {
                double E2 = Es[i1];
                double y2 = ys[i1];
                double ERatio = E2 / E1;
                double logEs = log( ERatio );
                double logYs = log( y2 / y1 );
                double _a = logYs / logEs;
                sum1 += integrateSub( 1,       _a, logEs, E1,      y1, E2,      y2 );
                sum2 += integrateSub( 1, 2.0 * _a, logEs, E1, y1 * y1, E2, y2 * y2 );
                std::cout << "                :: " << toString( xs[i1], xFmt ) << "  " << toString( E2, xFmt ) << "  " << toString( y2 ) << "  " << toString( sum1 ) << "  " << toString( sum2 ) << "  " << toString( _a ) << std::endl;
                E1 = E2;
                y1 = y2;
            }
        } }
    else {
        if( function->type( ) != GIDI::f_XYs1d ) {
            std::cout << "WARNING: unsupported data type = " << function->type( ) << std::endl;
            return;
        }

        GIDI::XYs1d const &xys1d = *static_cast<GIDI::XYs1d const *>( function );

        std::cout << "              interpolation = " << xys1d.interpolationString( ) << std::endl;
    }
}
/*
=========================================================
*/
double integrateSub( int _n, double _a, double logX, double E1, double _y1, double E2, double _y2 ) {

    double epsilon = _a + _n + 1;
    double integral;

    if( fabs( epsilon ) < 1e-3 ) {
        double epsilon_logX = epsilon * logX;
        integral = _y1 * pow( E1, _n + 1 ) * logX * ( 1 + 0.5 * epsilon_logX * ( 1 + epsilon_logX / 3.0 * ( 1 + 0.25 * epsilon_logX ) ) ); }
    else {
        integral = ( _y2 * pow( E2, _n + 1 ) - _y1 * pow( E1, _n + 1 ) ) / epsilon;
    }

    return( integral );
}
/*
=========================================================
*/
std::string toString( double a_value, char const *a_fmt ) {

    if( a_fmt == NULL ) a_fmt = " % 23.16e";
    char Str[128];
    sprintf( Str, a_fmt, a_value );
    return( std::string( Str ) );
}
