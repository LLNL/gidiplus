/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include <GIDI.hpp>

class argvOption {

    public:
        std::string m_name;     // Must include all leading '-'s (E.g., "-v", "--target").
        int m_counter;
        bool m_needsValue;
        std::vector<int> m_indices;
        std::string m_descriptor;

        argvOption( std::string const &a_name, bool a_needsValue, std::string const &a_descriptor = "" );

        bool present( ) const { return( m_counter > 0 ); }
        std::string zeroOrOneOption( char **argv, std::string const &a_default = "" );
        void help( );
        void print( );
};

class argvOptions {

    public:
        std::string m_codeName;
        std::string m_descriptor;
        int m_minimumNonOptions;
        std::vector<argvOption> m_options;
        std::vector<int> m_arguments;

        argvOptions( std::string const &a_codeName, std::string const &a_descriptor = "", int a_minimumNonOptions = 0 );

        int size( ) { return( static_cast<int>( m_options.size( ) ) ); }
        void add( argvOption const &a_option ) { m_options.push_back( a_option ); }
        void parseArgv( int argc, char **argv );
        argvOption *find( std::string const &a_name );
        void help( );
        void print( );
};

int outputChannelStringMaximumLength( GIDI::Protare const *protare );
std::string outputChannelString( GIDI::Reaction const *reaction );
std::string outputChannelPrefix( int offset, int width, GIDI::Reaction const *reaction );
long integerFromArgv( int iarg, int argc, char **argv );
void printVector( char const *prefix, GIDI::Vector &vector );
void printVector( std::string &prefix, GIDI::Vector &vector );
void printVectorOfDoubles( char const *a_prefix, std::vector<double> const &doubles );
void printVectorOfDoubles( std::string &a_prefix, std::vector<double> const &doubles );
void printIDs( char const *a_prefix, std::set<std::string> const &ids );
void printIDs( std::string &a_prefix, std::set<std::string> const &ids );
void printMatrix( char const *a_prefix, int maxOrder, GIDI::Matrix &matrix );
void printMatrix( std::string &prefix, int maxOrder, GIDI::Matrix &matrix );
std::string stripDirectoryBase( std::string const &a_path, std::string const &a_section = "/GIDI3/Test/map/neutrons/" );
std::string doubleToString( char const *format, double value );
