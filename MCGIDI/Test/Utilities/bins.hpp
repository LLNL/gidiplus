/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <vector>

/*
============================================================
=========================== Bins ===========================
============================================================
*/
class Bins {

    public:
        bool m_logDomainStep;
        double m_domainMin, m_domainMax, m_domainWidth;
        double m_logDomainFraction;
        long m_underFlows;
        long m_overFlows;
        std::vector<long> m_bins;
        double m_underFlowWeights;
        double m_overFlowWeights;
        std::vector<double> m_weightedBins;

        void setDomain( double a_domainMin, double a_domainMax ) {

            m_domainMin = a_domainMin;
            m_domainMax = a_domainMax;
            m_domainWidth = m_domainMax - m_domainMin;
            m_logDomainFraction = 2.0;
            if( m_logDomainStep ) m_logDomainFraction = log( pow( m_domainMax / m_domainMin, 1.0 / m_bins.size( ) ) );
        }

        Bins( long a_numberOfBins, double a_domainMin, double a_domainMax, bool a_logDomainStep = false ) :
                m_logDomainStep( a_logDomainStep ),
                m_underFlows( 0 ),
                m_overFlows( 0 ),
                m_bins( a_numberOfBins, 0 ),
                m_underFlowWeights( 0.0 ),
                m_overFlowWeights( 0.0 ),
                m_weightedBins( a_numberOfBins, 0.0 ) {

            setDomain( a_domainMin, a_domainMax );
        }

        void clear( ) {

            m_underFlows = 0;
            m_overFlows = 0;
            m_underFlowWeights = 0.0;
            m_overFlowWeights = 0.0;
            for( std::size_t i1 = 0; i1 < m_bins.size( ); ++i1 ) {
                m_bins[i1] = 0;
                m_weightedBins[i1] = 0.0;
            }
        }

        void accrue( double a_value, double a_weight = 0.0 ) {

            long index;

            if( m_logDomainStep ) {
                index = (long) ( log( a_value / m_domainMin ) / m_logDomainFraction ); }
            else {
                index = (long) ( ( a_value - m_domainMin ) / m_domainWidth * m_bins.size( ) );
            }

            if( index < 0 ) {
                ++m_underFlows;
                m_underFlowWeights += a_weight; }
            else if( index >= (long) m_bins.size( ) ) {
                ++m_overFlows;
                m_overFlowWeights += a_weight; }
            else {
                ++m_bins[index];
                m_weightedBins[index] += a_weight;
            }
        }

        long total( bool a_includeOutOfBounds ) {
            long sum = 0;

            if( a_includeOutOfBounds ) sum += m_underFlows + m_overFlows;

            for( std::size_t i1 = 0; i1 < m_bins.size( ); ++i1 ) sum +=  m_bins[i1];

            return( sum );
        }

        double totalWeights( bool a_includeOutOfBounds ) {
            double sum = 0;

            if( a_includeOutOfBounds ) sum += m_underFlowWeights + m_overFlowWeights;

            for( std::size_t i1 = 0; i1 < m_weightedBins.size( ); ++i1 ) sum +=  m_weightedBins[i1];

            return( sum );
        }

        double meanX( ) {

            double _total = total( false );

            if( _total == 0 ) return( 0.0 );

            double mean_x = 0.0;
            for( std::size_t i1 = 0; i1 <  m_bins.size( ); ++i1 ) {
                double x1;

                if( m_logDomainStep ) {
                    x1 = m_domainMin * exp( m_logDomainFraction * ( i1 + 0.5 ) ); }
                else {
                    x1 = ( i1 + 0.5 ) / ( (double) m_bins.size( ) ) * m_domainWidth + m_domainMin;
                }
                mean_x += m_bins[i1] * x1;
            }

            return( mean_x / _total );
        }

        void print( FILE *a_fOut, char const *a_label, bool a_includeWeights = false ) {

            long _total = total( false );
            double weightedTotal = totalWeights( false );

            fprintf( a_fOut, "\n\n" );
            if( strlen( a_label ) > 0 ) fprintf( a_fOut, "%s\n", a_label );
            fprintf( a_fOut, "# total number of inflows = %ld\n", _total );
            fprintf( a_fOut, "# number of underflows = %ld\n", m_underFlows );
            fprintf( a_fOut, "# number of overflows = %ld\n", m_overFlows );
            fprintf( a_fOut, "# number of Bins = %lu\n", m_bins.size( ) );
            fprintf( a_fOut, "# domain min = %g\n", m_domainMin );
            fprintf( a_fOut, "# domain max = %g\n", m_domainMax );
            if( a_includeWeights ) {
                fprintf( a_fOut, "# total weight = %15.7e\n", weightedTotal );
                fprintf( a_fOut, "# underflow weight = %15.7e\n", m_underFlowWeights );
                fprintf( a_fOut, "# overflow weight = %15.7e\n", m_overFlowWeights );
            }

            if( _total == 0 ) _total = 1;
            if( weightedTotal == 0.0 ) weightedTotal = 1.0;
            double norm = m_domainWidth / ( m_bins.size( ) + 1 );
            for( std::size_t i1 = 0; i1 <  m_bins.size( ); ++i1 ) {
                double x1;
                double partial = m_bins[i1] / (double) _total;

                if( m_logDomainStep ) {
                    x1 = m_domainMin * exp( m_logDomainFraction * ( i1 + 0.5 ) );
                    norm = x1 * ( exp( m_logDomainFraction ) - 1 ); }
                else {
                    x1 = ( i1 + 0.5 ) / ( (double) m_bins.size( ) ) * m_domainWidth + m_domainMin;
                }

                fprintf( a_fOut, "%23.15e  %15.7e  %8ld  %15.7e", x1, partial / norm, m_bins[i1], partial );
                if( a_includeWeights ) {
                    partial = m_weightedBins[i1] / weightedTotal;

                    fprintf( a_fOut, "  %15.7e  %15.7e  %15.7e",  partial / norm, m_weightedBins[i1], partial );
                }
                fprintf( a_fOut, "\n" );
            }
        }
};
