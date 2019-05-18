/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdarg.h>

#include <nfut_utilities.h>
#include <ptwXY.h>
#include <ptwXY_utilities.h>

static int verbose = 0;
char fmt[] = "%28.20e %28.20e\n";

int unitbase2s( statusMessageReporting *smr, int n1, double *xy1, int n2, double *xy2, double w, double w1, double w2 );
void printMsg( const char *fmt, ... );
/*
****************************************************************
*/
int main( int argc, char **argv ) {

    int iarg, echo = 0, errCount = 0;
    double xy1[3*2] = { 2.25028423, 0.0, 2.25028445, 4.54545455e+06, 2.25028467, 0.0};
    double xy2[3*2] = { 2.25211136, 0.0, 2.25211158, 4.54545455e+06,  2.25211180, 0.0 };
    double xy3[3*2] = { 2.60515525000000053302e-02, 0.0, 
                        2.60515547000000000089e-02, 4.54545454000000059605e+08, 
                        2.60515569000000016264e-02, 1.99999999591648003339e+00 };
    double xy4[7*2] = { 2.69831613500000067063e-02, 0.00000000000000000000e+00,
                        2.69831624500015167245e-02, 2.27272415168003588915e+08,
                        2.69831635500030267427e-02, 1.71142156489392403482e-03,
                        2.77848728987356886899e-02, 6.23665704927317278816e+02,
                        2.85865822499969807202e-02, 1.71142155949811892134e-03,
                        2.85865833499984942079e-02, 2.27272415168003588915e+08,
                        2.85865844500000007566e-02, 0.00000000000000000000e+00 };

    double xy5[3*2] = { 7.01724149e-05, 0.00000000e+00, 7.01724369e-05, 4.54545455e+10, 7.01724589e-05, 0.00000000e+00 };
    double xy6[3*2] = { 3.32286952e-03, 0.00000000e+00, 3.32286974e-03, 4.54545455e+09, 3.32286996e-03, 0.00000000e+00 };
    statusMessageReporting smr;

    smr_initialize( &smr, smr_status_Ok );

    for( iarg = 1; iarg < argc; iarg++ ) {
        if( strcmp( "-e", argv[iarg] ) == 0 ) {
            echo = 1; }
        else if( strcmp( "-v", argv[iarg] ) == 0 ) {
            verbose = 1; }
        else {
            printMsg( "Error %s: invalid input option '%s'", __FILE__, argv[iarg] );
        }
    }
    if( echo ) printf( "%s\n", __FILE__ );
    
    nfu_setMemoryDebugMode( 0 );

    errCount += unitbase2s( &smr, 3, xy1, 3, xy2, 0.5, 0., 1. );
    errCount += unitbase2s( &smr, 3, xy3, 7, xy4, -6.31237500000000006928e-01, -0.75, -0.625 );
    errCount += unitbase2s( &smr, 3, xy5, 3, xy6, -0.7689, -1, 0 );

    if( errCount ) fprintf( stderr, "ERROR in %s\n", __FILE__ );
    exit( errCount ? EXIT_FAILURE : EXIT_SUCCESS );
}
/*
****************************************************************
*/
int unitbase2s( statusMessageReporting *smr, int n1, double *xy1, int n2, double *xy2, double w, double w1, double w2 ) {

    int errCount = 0;
    int64_t i;
    ptwXYPoints *pXY1, *pXY2, *ub;
    double accuracy = 1e-3, x1, x2, y2;
    ptwXY_interpolation interpolation = ptwXY_interpolationLinLin;

    if( ( pXY1 = ptwXY_create( smr, interpolation, NULL, 5, accuracy, 10, 10, n1, xy1, 0 ) ) == NULL ) 
        nfut_printSMRErrorExit2p( smr, "Via." );
    if( ( pXY2 = ptwXY_create( smr, interpolation, NULL, 5, accuracy, 10, 10, n2, xy2, 0 ) ) == NULL ) 
        nfut_printSMRErrorExit2p( smr, "Via." );

    if( verbose ) {
        printf( "\n\n" );
        printf( "# length = %d\n", (int) pXY1->length );
        ptwXY_simpleWrite( pXY1, stdout, fmt );
        printf( "\n\n" );
        printf( "# length = %d\n", (int) pXY2->length );
        ptwXY_simpleWrite( pXY2, stdout, fmt );
        printf( "\n\n" );
        printf( "%e  %e  %e\n", w, w1, w2 );
        printf( "\n\n" );
    }

    if( ( ub = ptwXY_unitbaseInterpolate( smr, w, w1, pXY1, w2, pXY2, 1 ) ) == NULL )
        nfut_printSMRErrorExit2p( smr, "Via." );
    for( i = 0; i < ub->length; i++ ) {
        if( ptwXY_getXYPairAtIndex( smr, ub, i, &x2, &y2 ) != nfu_Okay ) nfut_printSMRErrorExit2p( smr, "Via." );
        if( i > 0 ) {
            if( verbose ) printf( "# x2 - x1 = %e\n", x2 - x1 );
            if( x2 == x1 ) ++errCount;
        }
        x1 = x2;
    }

    if( verbose ) {
        printf( "\n\n" );
        printf( "# length = %d\n", (int) ub->length );
        ptwXY_simpleWrite( ub, stdout, fmt );
    }

    ptwXY_free( pXY1 );
    ptwXY_free( pXY2 );
    ptwXY_free( ub );

    return( errCount );
}
/*
****************************************************************
*/
void printMsg( const char *fmt, ... ) {

    va_list args;

    va_start( args, fmt );
    vfprintf( stderr, fmt, args );
    fprintf( stderr, "\n" );
    va_end( args );
    exit( EXIT_FAILURE );
}
