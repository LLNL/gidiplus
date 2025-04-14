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

#include "GIDI.hpp"

static char const *description = "This test checks the pop method of the Suite class.";

/*
============================================================
=========================== Form ===========================
============================================================
*/

class Form : public GIDI::Form {

    public:
        Form( std::string a_label ) : GIDI::Form( "dummy", GIDI::FormType::generic, a_label ) { }
        ~Form( ) { std::cout << "    Deleting form '" << label( ) << "'" << std::endl; }
};

/*
============================================================
========================== Suite ==========================
============================================================
*/

class Suite : public GIDI::Suite {

    private:
        std::string m_name;

    public:
        Suite( std::string a_name ) { m_name = a_name; }
        ~Suite( ) { std::cout << std::endl << "Deleting suite " << m_name << " of size " << size( ) << std::endl; }

        std::string name( ) { return( m_name ); }
};

void main2( int argc, char **argv );
void printSuiteLabls( Suite &a_suite );
void pop( Suite &a_suite1, Suite &a_suite2, std::size_t );
void pop( Suite &a_suite1, Suite &a_suite2, std::string a_label );

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

    LUPI::ArgumentParser argumentParser( __FILE__, description );
    argumentParser.parse( argc, argv );

    Suite suite1( "1" );
    Suite suite2( "2" );

    suite1.add( new Form( "Hello" ) );
    suite1.add( new Form( "world" ) );
    suite1.add( new Form( "I" ) );
    suite1.add( new Form( "am" ) );
    suite1.add( new Form( "going." ) );
    suite1.add( new Form( "Good" ) );
    suite1.add( new Form( "bye" ) );

    printSuiteLabls( suite1 );

    pop( suite1, suite2, 2 );
    pop( suite1, suite2, 4 );
    pop( suite1, suite2, "bye" );
    pop( suite1, suite2, 0 );
}

/*
=========================================================
*/

void printSuiteLabls( Suite &a_suite ) {

    std::cout << std::endl;
    std::cout << "Printing suite " << a_suite.name( ) << std::endl;
    for( auto iter = a_suite.begin( ); iter != a_suite.end( ); ++iter ) {
        std::cout << "    " << (*iter)->label( ) << std::endl;
    }
}

/*
=========================================================
*/

void pop( Suite &a_suite1, Suite &a_suite2, std::size_t a_index ) {

    std::cout << std::endl;

    Form const *form = a_suite1.get<Form>( a_index );
    std::cout << "Poping at index " << a_index << " with label '" << form->label( ) << "'." << std::endl;
    a_suite2.add( a_suite1.pop<Form>( a_index ) );
}

/*
=========================================================
*/

void pop( Suite &a_suite1, Suite &a_suite2, std::string a_label ) {

    std::cout << std::endl;

    std::cout << "Poping label '" << a_label << std::endl;
    a_suite2.add( a_suite1.pop<Form>( a_label ) );
}
