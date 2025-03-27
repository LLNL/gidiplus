/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include <iostream>
#ifdef GIDI_PLUS_INCLUDE_PYTHON_BUILDING
#include <Python.h>
#endif

#include <GIDIP_python.hpp>

namespace GIDIP {

namespace Python {

#ifdef GIDI_PLUS_INCLUDE_PYTHON_BUILDING
#include <Python.h>

/* *********************************************************************************************************//**
 * This function loads the python module referenced by **a_moduleName** and returns it. The argument **a_moduleName**
 * must be the path to the module to load but without the file's extension. For example, to load the module with path
 * "path/to/module/myModule.py", **a_moduleName** should be "path/to/module/myModule". Currently, this function
 * only works for ".py" extensions.
 *
 * The caller owns the reference to the returned object. Ergo, the caller is responsible for calling *Py_DECREF* on the returned object.
 *
 * A throw is executed if the module does not exist.
 *
 * @param       a_moduleName        [in]    The path to the module but without the extension.
 *
 * @returns                                 A pointer to an instance of PyObject.
 ***********************************************************************************************************/

void *loadModule( std::string const &a_moduleName ) {

    if( !Py_IsInitialized( ) ) Py_Initialize( );

    PyObject *py_sys_path = PySys_GetObject( "path" );

    bool found = false;
    std::string dirPath = LUPI::FileInfo::_dirname( LUPI::FileInfo::realPath( a_moduleName + ".py" ) );
    auto py_sys_pathSize = PyList_GET_SIZE( py_sys_path );
    auto index = py_sys_pathSize;
    for( index = 0; index < py_sys_pathSize; ++index ) {
        PyObject *py_pathItem = PyList_GET_ITEM( py_sys_path, index );
        PyObject *py_pathItemASCII = PyUnicode_AsASCIIString( py_pathItem );
        std::string pathItem( PyBytes_AS_STRING( py_pathItemASCII ) );
        if( LUPI::FileInfo::exists( pathItem ) ) {
            if( LUPI::FileInfo::realPath( pathItem ) == dirPath ) found = true;
        }
    }
    if( !found ) {
        PyObject *py_path = PyUnicode_FromString( dirPath.c_str( ) );
        PyList_Insert( py_sys_path, 0, py_path );
    }

    std::string baseModule = LUPI::FileInfo::_basename( a_moduleName );
    PyObject *py_module = PyImport_ImportModule( a_moduleName.c_str( ) );

    return( py_module );
}

/* *********************************************************************************************************//**
 * This function returns a *void* pointer which is a *PyObject* pointer to the function named **a_functionName** 
 * in module **a_moduleName**.  Also see function *loadModule*. The caller owns the reference to the returned object. 
 * Ergo, the caller is responsible for calling *Py_DECREF* on the returned object.
 *
 * A throw is executed if the module does not exist. A *nullptr* is returned if the function does not exist in the
 * module.
 *
 * @param       a_moduleName        [in]    The path to the module but without the extension.
 * @param       a_functionName      [in]    The name of the function in the module whose reference is returned.
 *
 * @returns                                 A pointer to an instance of PyObject.
 ***********************************************************************************************************/

void *loadFunctionInModule( std::string const &a_moduleName, std::string const &a_functionName ) {

    PyObject *py_function = nullptr;
    PyObject *py_module = static_cast<PyObject *>( loadModule( a_moduleName ) );

    if( PyObject_HasAttrString( py_module, a_functionName.c_str( ) ) > 0 ) {
        py_function = PyObject_GetAttrString( py_module, a_functionName.c_str( ) );
        if( !PyCallable_Check( py_function ) ) {
            Py_DECREF( py_function );
            py_function = nullptr;
        }
    }

    Py_DECREF( py_module );

    return( py_function );
}

/* *********************************************************************************************************//**
 * This function calls *Py_DECREF* on **a_pyObject** which must be a pointer to a *PyObject* instance. If
 * **a_pyObject** is a nullptr, *Py_DECREF* is not called.
 *
 * @param       a_pyObject      [in]    A pointer to a *PyObject* instance which the arguemnt for a call to *Py_DECREF*.
 ***********************************************************************************************************/

void decrementRef( void *a_pyObject ) {

    if( a_pyObject != nullptr ) Py_DECREF( static_cast<PyObject *>( a_pyObject ) );
}

/* *********************************************************************************************************//**
 * This function calls a the python function **a_PyFunction** with argument **a_PyArgs**, converts its return value to an
 *
 * @param       a_PyFunction        [in]    A pointer to a *PyObject* instance that is the python function to call.
 * @param       a_PyArgs             [in]    Either *nullptr* or a pointer to a Python tuple representing the argument for **a_PyFunction**.
 *
 * @returns                                 A pointer to an instance of PyObject.
 ***********************************************************************************************************/

double callFunctionReturnDouble( void *a_PyFunction, void *a_PyArgs ) {

    PyObject *py_function = static_cast<PyObject *>( a_PyFunction );
    PyObject *py_args = static_cast<PyObject *>( a_PyArgs );
    PyObject *py_value = PyObject_CallObject( py_function, py_args );
    if( py_value == nullptr ) throw LUPI::Exception( "Python function call failed." );

    double value = PyFloat_AsDouble( py_value );
    Py_DECREF( py_value );

    return( value );
}

#else

//
// If python stuff not to be included, define functions to only execute a throw.
//

void *loadModule( LUPI_maybeUnused std::string const &a_moduleName ) {
    throw LUPI::Exception( "GIDIP::Python::loadModule: python build not included." );
    return( nullptr );
}

void *loadFunctionInModule( LUPI_maybeUnused std::string const &a_moduleName, LUPI_maybeUnused std::string const &a_functionName ) {
    throw LUPI::Exception( "GIDIP::Python::loadFunctionInModule: python build not included." );
    return( nullptr );
}

void decrementRef( LUPI_maybeUnused void *a_pyObject ) {
    throw LUPI::Exception( "GIDIP::Python::decrementRef: python build not included." );
}

double callFunctionReturnDouble( LUPI_maybeUnused void *a_PyFunction, LUPI_maybeUnused void *a_PyArgs ) {
    throw LUPI::Exception( "GIDIP::Python::callFunctionReturnDouble: python build not included." );
    return( 0.0 );
}

#endif
}               // End of namespace Python.

}               // End of namespace GIDIP.
