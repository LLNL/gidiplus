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
char fmt[] = "%22.14e %22.14e\n";

static void printUnitbasedXY( double w, double wMin, double wMax, ptwXYPoints *p );
void printMsg( const char *fmt, ... );
/*
****************************************************************
*/
int main( int argc, char **argv ) {

    int64_t i;
    int iarg, echo = 0, errCount = 0;
    ptwXYPoints *pXY1, *pXY2, *pl, *pr, *pm1, *pm2, *diff;
    ptwXYPoint *p;
    double y, accuracy = 1e-3, xy1[3*2] = { -1., 0., 0., 1., 1., 0. }, xy2[3*2] = { 8., 0., 10.5, 0.4, 13., 0. };
    nfu_status status;
    ptwXY_interpolation interpolation = ptwXY_interpolationLinLin;
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

    if( ( pXY1 = ptwXY_create( &smr, interpolation, NULL, 5, accuracy, 10, 10,    3, xy1, 0 ) ) == NULL ) 
        nfut_printSMRErrorExit2p( &smr, "Via." );
    if( ( pXY2 = ptwXY_create( &smr, interpolation, NULL, 5, accuracy, 10, 10,    3, xy2, 0 ) ) == NULL ) 
        nfut_printSMRErrorExit2p( &smr, "Via." );

    if( ( pl = ptwXY_unitbaseInterpolate( &smr, 4., 0., pXY1, 20., pXY2, 1 ) ) == NULL )
        nfut_printSMRErrorExit2p( &smr, "Via." );

    if( ( pr = ptwXY_unitbaseInterpolate( &smr, 12., 0., pXY1, 20., pXY2, 1 ) ) == NULL )
        nfut_printSMRErrorExit2p( &smr, "Via." );
    if( ( pm1 = ptwXY_unitbaseInterpolate( &smr, 10., 4., pl, 12., pr, 1 ) ) == NULL )
        nfut_printSMRErrorExit2p( &smr, "Via." );

    if( ( pm2 = ptwXY_unitbaseInterpolate( &smr, 10., 0., pXY1, 20., pXY2, 1 ) ) == NULL )
        nfut_printSMRErrorExit2p( &smr, "Via." );

    if( ( diff = ptwXY_sub_ptwXY( &smr, pm1, pm2 ) ) == NULL ) nfut_printSMRErrorExit2p( &smr, "Via." );
    for( i = 0; i < ptwXY_length( &smr, diff ); i++ ) {
        p = ptwXY_getPointAtIndex_Unsafely( diff, i );
        if( ( status = ptwXY_getValueAtX( &smr, pm1, p->x, &y ) ) != nfu_Okay ) nfut_printSMRErrorExit2p( &smr, "Via." );
        switch( status ) {
        case nfu_Okay :
            if( fabs( p->y ) > 1e-12 * fabs( y ) )
                printMsg( "pm1 and pm2 differ at x  = %e by %e:, pm1.y = %e", p->x, p->y, y );
            break;
        case nfu_XOutsideDomain :
            if( ( i == 0 ) || ( i == ( ptwXY_length( &smr, diff ) - 1 ) ) ) continue;
            printMsg( "ptwXY_getValueAtX status = %d: %s", status, nfu_statusMessage( status ) );
        default :
            printMsg( "ptwXY_getValueAtX status = %d: %s", status, nfu_statusMessage( status ) );
        }
    }
    ptwXY_free( diff );

    if( verbose ) {
        printf( "\n\n" );
        printf( "# length = %d\n", (int) pXY1->length );
        ptwXY_simpleWrite( pXY1, stdout, fmt );
        printf( "\n\n" );
        printf( "# length = %d\n", (int) pXY2->length );
        ptwXY_simpleWrite( pXY2, stdout, fmt );
        printUnitbasedXY(  4., 0., 20., pl );
        printUnitbasedXY( 12., 0., 20., pr );
        printUnitbasedXY( 10., 4., 12., pm1 );
        printUnitbasedXY( 10., 0., 20., pm2 );
    }

    ptwXY_free( pXY1 );
    ptwXY_free( pXY2 );
    ptwXY_free( pl );
    ptwXY_free( pr );
    ptwXY_free( pm1 );
    ptwXY_free( pm2 );

    exit( errCount ? EXIT_FAILURE : EXIT_SUCCESS );
}
/*
****************************************************************
*/
static void printUnitbasedXY( double w, double wMin, double wMax, ptwXYPoints *p ) {

    printf( "\n\n" );
    printf( "# w = %e\n", w );
    printf( "# wMin = %e\n", wMin );
    printf( "# wMax = %e\n", wMax );
    printf( "# length = %d\n", (int) p->length );
    ptwXY_simpleWrite( p, stdout, fmt );
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
