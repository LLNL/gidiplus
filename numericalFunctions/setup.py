# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

import os, glob, shutil
from distutils.core import setup, Extension

extra_compile_args = [ ]
# Option need for some MACs
# extra_compile_args = [ '-Wno-error=unused-command-line-argument-hard-error-in-future' ]

statusMessageReportingRoot = '../statusMessageReporting'

statusMessageReporting_c = glob.glob( os.path.join( statusMessageReportingRoot, 'Src', '*.c' ) )
ptwC_c = glob.glob( os.path.join( 'ptwC', 'Src', '*.c' ) )
ptwX_c = glob.glob( os.path.join( 'ptwX', 'Src', '*.c' ) )
ptwX_Py_c = glob.glob( os.path.join( 'ptwX', 'Python', 'Src', '*.c' ) )
ptwXY_c = glob.glob( os.path.join( 'ptwXY', 'Src', '*.c' ) )
ptwXY_Py_c = glob.glob( os.path.join( 'ptwXY', 'Python', 'Src', '*.c' ) )
nf_Legendre_c = glob.glob( os.path.join( 'nf_Legendre', 'Src', '*.c' ) )
nf_Legendre_Py_c = glob.glob( os.path.join( 'nf_Legendre', 'Python', 'Src', '*.c' ) )
nf_specialFunctions_c = glob.glob( os.path.join( 'nf_specialFunctions', 'Src', 'nf_[egip]*.c' ) )
nf_specialFunctions_Py_c = glob.glob( os.path.join( 'nf_specialFunctions', 'Python', 'Src', 'nf_specialFunctions_C.c' ) )
nf_angularMomentumCoupling_c = glob.glob( os.path.join( 'nf_specialFunctions', 'Src', 'nf_angularMomentumCoupling.c' ) )
nf_angularMomentumCoupling_Py_c = glob.glob( os.path.join( 'nf_specialFunctions', 'Python', 'Src', 'nf_angularMomentumCoupling_C.c' ) )
nf_integration_c = glob.glob( os.path.join( 'nf_integration', 'Src', '*.c' ) )
nf_integration_Py_c = glob.glob( os.path.join( 'nf_integration', 'Python', 'Src', '*.c' ) )

statusMessageReporting_hDir = os.path.join( statusMessageReportingRoot, 'Src' )
ptwC_hDir = os.path.join( 'ptwC', 'Src' )
ptwX_hDir = os.path.join( 'ptwX', 'Src' )
ptwXY_hDir = os.path.join( 'ptwXY', 'Src' )
ptwXY_Py_hDir =  os.path.join( 'ptwXY', 'Python', 'Src' )
nf_Legendre_hDir = os.path.join( 'nf_Legendre', 'Src' )
nf_specialFunctions_hDir = os.path.join( 'nf_specialFunctions', 'Src' )
nf_specialFunctions_Py_hDir = os.path.join( 'nf_specialFunctions', 'Python', 'Src' )
nf_integration_hDir = os.path.join( 'nf_integration', 'Src' )

#
# Stuff to build listOfDoubles.so.
#
libs = glob.glob( os.path.join( 'build', 'lib*', 'listOfDoubles_C*' ) )
for lib in libs : os.remove( lib )

listOfDoubles_C = Extension( 'listOfDoubles_C',
    extra_compile_args = extra_compile_args,
    sources = statusMessageReporting_c + ptwC_c + ptwX_c + ptwX_Py_c,
    include_dirs = [ statusMessageReporting_hDir, ptwC_hDir, ptwX_hDir ] )

setup( name = 'listOfDoubles_C', 
    version = '1.0',
    description = 'This module contains the listOfDoubles_C class and support routines.',
    ext_modules = [ listOfDoubles_C ] )

libs = glob.glob( os.path.join( 'build', 'lib*', 'listOfDoubles_C*' ) )
if( len( libs ) > 0 ) : shutil.copy( libs[0], 'lib' )

#
# Stuff to build pointwiseXY_C.so.
#
libs = glob.glob( os.path.join( 'build', 'lib*', 'pointwiseXY_C*' ) )
for lib in libs : os.remove( lib )

pointwiseXY_C = Extension( 'pointwiseXY_C',
    extra_compile_args = extra_compile_args,
    sources = statusMessageReporting_c + ptwC_c + ptwX_c + nf_Legendre_c + nf_integration_c + ptwXY_c + ptwXY_Py_c,
    include_dirs = [ statusMessageReporting_hDir, ptwC_hDir, ptwX_hDir, nf_Legendre_hDir, nf_integration_hDir, 
            ptwXY_hDir, ptwXY_Py_hDir ] )

setup( name = 'pointwiseXY_C', 
    version = '1.0',
    description = 'This module contains the pointwiseXY_C class and support routines.',
    ext_modules = [ pointwiseXY_C ] )

libs = glob.glob( os.path.join( 'build', 'lib*', 'pointwiseXY_C*' ) )
if( len( libs ) > 0 ) : shutil.copy( libs[0], 'lib' )

#
# Stuff to build Legendre.so.
#
libs = glob.glob( os.path.join( 'build', 'lib*', 'Legendre*' ) )
for lib in libs : os.remove( lib )

nf_Legendre_C = Extension( 'Legendre',
    extra_compile_args = extra_compile_args,
    sources = statusMessageReporting_c + ptwC_c + ptwX_c + ptwXY_c + nf_integration_c + nf_Legendre_c + nf_Legendre_Py_c,
    include_dirs = [ statusMessageReporting_hDir, ptwC_hDir, ptwX_hDir, ptwXY_hDir, ptwXY_Py_hDir, 
            nf_integration_hDir, nf_Legendre_hDir ] )

setup( name = 'Legendre', 
    version = '1.0',
    description = 'This module contains the Legendre class and support routines.',
    ext_modules = [ nf_Legendre_C ] )

libs = glob.glob( os.path.join( 'build', 'lib*', 'Legendre*' ) )
if( len( libs ) > 0 ) : shutil.copy( libs[0], 'lib' )

#
# Stuff to build specialFunctions.so
#
libs = glob.glob( os.path.join( 'build', 'lib*', 'specialFunctions*' ) )
for lib in libs : os.remove( lib )

specialFunctions = Extension( 'specialFunctions',
    extra_compile_args = extra_compile_args,
    sources = statusMessageReporting_c + ptwC_c + nf_specialFunctions_c + nf_specialFunctions_Py_c,
    include_dirs = [ statusMessageReporting_hDir, ptwC_hDir, nf_specialFunctions_hDir, nf_specialFunctions_Py_hDir ] )

setup( name = 'specialFunctions', 
    version = '1.0',
    description = 'This module contains some special math functions not in the python math module.',
    ext_modules = [ specialFunctions ] )

libs = glob.glob( os.path.join( 'build', 'lib*', 'specialFunctions*' ) )
if( len( libs ) > 0 ) : shutil.copy( libs[0], 'lib' )

#
# Stuff to build angularMomentumCoupling.so
#
libs = glob.glob( os.path.join( 'build', 'lib*', 'angularMomentumCoupling*' ) )
for lib in libs : os.remove( lib )

angularMomentumCoupling = Extension( 'angularMomentumCoupling',
    extra_compile_args = extra_compile_args,
    sources = statusMessageReporting_c + ptwC_c + nf_angularMomentumCoupling_c + nf_angularMomentumCoupling_Py_c ,
    include_dirs = [ statusMessageReporting_hDir, ptwC_hDir, nf_specialFunctions_hDir, nf_specialFunctions_Py_hDir ] )

setup( name = 'angularMomentumCoupling', 
    version = '1.0',
    description = 'This module contains some physics angular momentum coupling functions not in the python math module.',
    ext_modules = [ angularMomentumCoupling ] )

libs = glob.glob( os.path.join( 'build', 'lib*', 'angularMomentumCoupling*' ) )
if( len( libs ) > 0 ) : shutil.copy( libs[0], 'lib' )

#
# Stuff to build integration.so
#
libs = glob.glob( os.path.join( 'build', 'lib*', 'integration*' ) )
for lib in libs : os.remove( lib )

integration = Extension( 'integration',
    extra_compile_args = extra_compile_args,
    sources = statusMessageReporting_c + ptwC_c + ptwX_c + ptwXY_c + nf_Legendre_c + nf_integration_c + nf_integration_Py_c,
    include_dirs = [ statusMessageReporting_hDir, ptwC_hDir, ptwX_hDir, ptwXY_hDir, nf_Legendre_hDir, nf_integration_hDir ] )

setup( name = 'integration', 
    version = '1.0',
    description = 'This module contains functions for integrating a function representing an integrand.',
    ext_modules = [ integration ] )

libs = glob.glob( os.path.join( 'build', 'lib*', 'integration*' ) )
if( len( libs ) > 0 ) : shutil.copy( libs[0], 'lib' )
