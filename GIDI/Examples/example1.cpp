/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include <stdlib.h>
#include <iostream>
#include <iomanip>

#include <GIDI_testUtilities.hpp>

void main2( int argc, char **argv );
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

                                                //! [snippet]
    PoPI::Database pops( "../../TestData/PoPs/pops.xml" );
    GIDI::Map::Map map( "../Test/all.map", pops );

    GIDI::Construction::Settings construction( GIDI::Construction::ParseMode::all,
                                               GIDI::Construction::PhotoMode::nuclearAndAtomic );
    GIDI::ProtareSingle *protare = static_cast<GIDI::ProtareSingle *>( map.protare( construction, pops, "n", "O16" ) );

    GIDI::Reaction *reaction = protare->reactions( ).get<GIDI::Reaction>( 1 );
    std::cout << "Reaction label is '" << reaction->label( ) << "'" << std::endl;

    GIDI::Suite &crossSection = reaction->crossSection( );
    GIDI::Functions::Function1dForm *form = crossSection.get<GIDI::Functions::Function1dForm>( "heated_000" );
    std::cout << "    cross section at 7.1 MeV is " << form->evaluate( 7.1 ) << std::endl;

    std::cout << std::endl;
    GIDI::OutputChannel *outputChannel = reaction->outputChannel( );
    GIDI::Suite &products = outputChannel->products( );
    std::cout << "    List of products for this reaction is" << std::endl;
    for( auto productIter = products.begin( ); productIter != products.end( ); ++productIter ) {
        GIDI::Product *product = static_cast<GIDI::Product *>( *productIter );

        std::cout << "        product id = '" << product->particle( ).ID( ) << "' label = '" << product->label( ) << "'" << std::endl;
    }
    
    GIDI::Product *product = products.get<GIDI::Product>( "n" );
    std::cout << std::endl;
    std::cout << "    List of distribution forms for product with id = '" <<
            product->particle( ).ID( ) << "' label = '" << product->label( ) << "' is" << std::endl;
    GIDI::Suite &distribution = product->distribution( );
    for( auto formIter = distribution.begin( ); formIter != distribution.end( ); ++formIter ) {
        std::cout << "            " << (*formIter)->moniker( ) << "  " << (*formIter)->label( ) << std::endl;
    }

    delete protare;
                                                //! [snippet]
}
