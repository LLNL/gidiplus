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

#include "GIDI_testUtilities.hpp"

static char const *description = "This program prints the multi-group available energy for a protare and its reactions.";

void main2( int argc, char **argv );
void readProtare( GIDI::Map::Map &map, PoPI::Database const &pops, std::string const &targetID );
void printFunctionInfo( std::string const &name, GIDI::Functions::Function1dForm *function );
void addInterpolations( ptwXY_interpolation a_interpolation, std::vector<ptwXY_interpolation> &a_interpolations, std::size_t a_size );
double integrateSub( int _n, double _a, double logE, double E1, double _y1, double E2, double _y2 );
std::string toString( double a_value, char const *a_fmt = nullptr );
/*
=========================================================
*/
int main( int argc, char **argv ) {

    try {
        main2( argc, argv ); }
     catch (std::exception &exception) {
        std::cerr << exception.what( ) << std::endl;
        exit( EXIT_FAILURE ); }
    catch (char const *str) {
        std::cerr << str << std::endl;
        exit( EXIT_FAILURE ); }
    catch (std::string &str) {
        std::cerr << str << std::endl;
        exit( EXIT_FAILURE );
    }

    exit( EXIT_SUCCESS );
}
/*
=========================================================
*/
void main2( int argc, char **argv ) {

    printCodeArguments( __FILE__, argc, argv );

    argvOptions argv_options( "coherentPhotonScattering", description );

    argv_options.add( argvOption( "--map", true, "The map file to use." ) );
    argv_options.parseArgv( argc, argv );

    PoPI::Database pops( "../../../TestData/PoPs/pops.xml" );
    GIDI::Map::Map map( argv_options.find( "--map" )->zeroOrOneOption( argv, "../all.map" ), pops );

    readProtare( map, pops, "O16" );
    readProtare( map, pops, "Th229" );
}
/*
=========================================================
*/
void readProtare( GIDI::Map::Map &map, PoPI::Database const &pops, std::string const &targetID ) {

    GIDI::Construction::Settings construction( GIDI::Construction::ParseMode::all, GIDI::Construction::PhotoMode::atomicOnly );
    GIDI::Protare *protare = map.protare( construction, pops, PoPI::IDs::photon, targetID );
    if( protare == nullptr ) {
        std::cout << "protare for " << targetID << " not found." << std::endl;
        exit( EXIT_FAILURE );
    }

    std::cout << stripDirectoryBase( protare->fileName( ), "/GIDI/Test/" ) << std::endl;

    for( std::size_t index = 0; index < protare->numberOfReactions( ); ++index ) {
        GIDI::Reaction *reaction = protare->reaction( index );
        GIDI::OutputChannel *outputChannel = reaction->outputChannel( );
        GIDI::Product *product = static_cast<GIDI::Product *>( outputChannel->products( ).get<GIDI::Product>( 0 ) );
        GIDI::Suite &distribution = product->distribution( );

        std::cout << "    " << reaction->label( ) << std::endl;
        std::cout << "      " << product->particle( ).ID( ) << std::endl;

        for( std::size_t i2 = 0; i2 < distribution.size( ); ++i2 ) {
            GIDI::Distributions::Distribution *form = distribution.get<GIDI::Distributions::Distribution>( i2 );

            std::cout << "        distribution form moniker = " << form->moniker( ) << std::endl;

            if( form->moniker( ) == GIDI_coherentPhotonScatteringChars ) {
                GIDI::Distributions::CoherentPhotoAtomicScattering *coherentPhotonScattering = static_cast<GIDI::Distributions::CoherentPhotoAtomicScattering *>( form );

                std::cout << "          href = " << coherentPhotonScattering->href( ) << std::endl;

                GUPI::Ancestry *link = coherentPhotonScattering->findInAncestry( coherentPhotonScattering->href( ) );

                GIDI::DoubleDifferentialCrossSection::CoherentPhotoAtomicScattering *dd = static_cast<GIDI::DoubleDifferentialCrossSection::CoherentPhotoAtomicScattering *>( link );
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
void printFunctionInfo( std::string const &name, GIDI::Functions::Function1dForm *function ) {

    char xFmt[] = "%23.16e";

    if( function == nullptr ) return;

    std::string domainUnit = function->axes( )[0]->unit( );
    std::string rangeUnit = function->axes( )[1]->unit( );
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
        std::vector<ptwXY_interpolation> interpolations;
        std::vector<double> xs;
        std::vector<double> ys;

        if( function->type( ) == GIDI::FormType::XYs1d ) {
            GIDI::Functions::XYs1d &xys1d = *static_cast<GIDI::Functions::XYs1d *>( function );

            std::cout << "              interpolation = " << xys1d.interpolationString( ) << std::endl;
            xs = xys1d.xs( );
            ys = xys1d.ys( );
            addInterpolations( xys1d.interpolation( ), interpolations, xys1d.xs( ).size( ) ); }
        else if( function->type( ) == GIDI::FormType::regions1d ) {
            GIDI::Functions::Regions1d *regions1d = (GIDI::Functions::Regions1d *)( function );
            for( std::size_t i1 = 0; i1 < regions1d->Xs( ).size( ); ++i1 ) std::cout << "              x[" << i1 << "] = " << regions1d->Xs( )[i1] << std::endl;

            std::vector<GIDI::Functions::Function1dForm *> &functions1d = regions1d->function1ds( );
            std::cout << "              number of regions = " << functions1d.size( );
            for( std::size_t i1 = 0; i1 < functions1d.size( ); ++i1 ) {
                GIDI::Functions::XYs1d &xys1d = *static_cast<GIDI::Functions::XYs1d *>( functions1d[i1] );

                std::cout << "              interpolation = " << xys1d.interpolationString( ) << std::endl;
            }
            if( functions1d.size( ) != 2 ) {
                std::cout << "WARNING: unsupported number of regions in " << function->moniker( ) << std::endl;
                return;
            }
            GIDI::Functions::XYs1d &xys1d1 = *static_cast<GIDI::Functions::XYs1d *>( functions1d[0] );
            xs = xys1d1.xs( );
            ys = xys1d1.ys( );
            addInterpolations( xys1d1.interpolation( ), interpolations, xys1d1.xs( ).size( ) );

            GIDI::Functions::XYs1d &xys1d2 = *static_cast<GIDI::Functions::XYs1d *>( functions1d[1] );
            auto xs2 = xys1d2.xs( );
            auto ys2 = xys1d2.ys( );
            addInterpolations( xys1d2.interpolation( ), interpolations, xys1d2.xs( ).size( ) );
            for( std::size_t index = 0; index < xs2.size( ); ++index ) {
                xs.push_back( xs2[index] );
                ys.push_back( ys2[index] );
            } }
        else {
            std::cout << "WARNING: unsupported data type = " << function->moniker( ) << std::endl;
            return;
        }

        std::vector<double> Es( xs );
        for( std::size_t i1 = 0; i1 < Es.size( ); ++i1 ) Es[i1] *= domainFactor;
        double x1 = xs[0], E1 = Es[0], y1 = ys[0];
        double sum1 = 0.0, sum2 = 0.0, _a = 0.0;
        std::cout << "                :: " << toString( xs[0], xFmt ) << "  " << toString( E1, xFmt ) << "  " 
                << toString( y1 ) << "  " << toString( sum1 ) << "  " << toString( sum2 ) << std::endl;
        for( std::size_t i1 = 1; i1 < xs.size( ); ++i1 ) {
            double x2 = xs[i1];
            double E2 = Es[i1];
            double y2 = ys[i1];
            if( x1 != x2 ) {
                if( interpolations[i1] == ptwXY_interpolationLinLin ) {
                    sum1 += ( E2 - E1 ) * ( y1 * ( 2 * E1 + E2 ) + y2 * ( E1 + 2 * E2 ) ) / 6;
                    sum2 += ( E2 - E1 ) * ( 2 * E1 * y1 * y1 + 2 * E2 * y2 * y2 + ( E1 + E2 ) * ( y1 + y2 ) * ( y1 + y2 ) ) / 12; }
                else if( interpolations[i1] == ptwXY_interpolationLogLog ) {
                    double ERatio = E2 / E1;
                    double logEs = log( ERatio );
                    double logYs = log( y2 / y1 );
                    _a = logYs / logEs;
                    sum1 += integrateSub( 1,       _a, logEs, E1,      y1, E2,      y2 );
                    sum2 += integrateSub( 1, 2.0 * _a, logEs, E1, y1 * y1, E2, y2 * y2 );
                }
                std::cout << "                :: " << toString( x2, xFmt ) << "  " << toString( E2, xFmt ) << "  " 
                        << toString( y2 ) << "  " << toString( sum1 ) << "  " << toString( sum2 );
                if( interpolations[i1] == ptwXY_interpolationLogLog ) {
                    std::cout << "  " << toString( _a );
                }
                std::cout << std::endl;
            }
            x1 = x2;
            E1 = E2;
            y1 = y2;
        } }
    else {
        if( function->type( ) != GIDI::FormType::XYs1d ) {
            std::cout << "WARNING: unsupported data type = " << function->moniker( ) << std::endl;
            return;
        }

        GIDI::Functions::XYs1d &xys1d = *static_cast<GIDI::Functions::XYs1d *>( function );

        std::cout << "              interpolation = " << xys1d.interpolationString( ) << std::endl;
    }
}
/*
=========================================================
*/
void addInterpolations( ptwXY_interpolation a_interpolation, std::vector<ptwXY_interpolation> &a_interpolations, std::size_t a_size ) {

    for( std::size_t index = 0; index < a_size; ++index ) {
        a_interpolations.push_back( a_interpolation );
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

    if( a_fmt == nullptr ) a_fmt = " % 23.16e";
    return( LUPI::Misc::argumentsToString( a_fmt, a_value ) );
}
