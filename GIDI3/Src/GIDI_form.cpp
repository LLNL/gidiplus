/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include <stdlib.h>
#include "GIDI.hpp"

namespace GIDI {

/*! \class Form 
 * Base class inherited by most other GIDI classes. Mainly contains **label** and **type** members.
 */

/* *********************************************************************************************************//**
 *
 * @param a_type            [in]    The *formType* the class represents.
 ***********************************************************************************************************/

Form::Form( formType a_type ) :
        Ancestry( "" ),
        m_parent( NULL ),
        m_type( a_type ),
        m_label( "" ) {

}

/* *********************************************************************************************************//**
 *
 * @param a_type            [in]    The *formType* the class represents.
 * @param a_label           [in]    The label for *this*.
 ***********************************************************************************************************/

Form::Form( formType a_type, std::string const &a_label ) :
        Ancestry( "" ),
        m_parent( NULL ),
        m_type( a_type ),
        m_label( a_label ) {

}

/* *********************************************************************************************************//**
 *
 * @param a_node            [in]    The **pugi::xml_node** to be parsed and used to construct the XYs2d.
 * @param a_type            [in]    The *formType* the class represents.
 * @param a_parent          [in]    The parent GIDI::Suite.
 ***********************************************************************************************************/

Form::Form( pugi::xml_node const &a_node, formType a_type, Suite *a_parent ) :
        Ancestry( a_node.name( ) ),
        m_parent( a_parent ),
        m_type( a_type ),
        m_label( a_node.attribute( "label" ).value( ) ) {

}

/* *********************************************************************************************************//**
 * @param a_form            [in]    The Form to copy.
 ***********************************************************************************************************/

Form::Form( Form const &a_form ) :
        Ancestry( a_form.moniker( ) ),
        m_parent( NULL ),
        m_type( a_form.type( ) ),
        m_label( a_form.label( ) ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

Form::~Form( ) {

}

/* *********************************************************************************************************//**
 * Returns the sibling of *this* with label *a_label*.
 *
 * @param a_label           [in]    The label of the sibling to find.
 * @return                          The sibling with label *a_label*.
 ***********************************************************************************************************/

Form const *Form::sibling( std::string a_label ) const {

    Form const *_form;

    try {
        _form = ((*parent( )).get<Form>( a_label ) ); }
    catch (...) {
        return( NULL );
    }
    return( _form );
}

/*! \class FunctionForm
 * Base class inherited by other GIDI function classes.
 */

/* *********************************************************************************************************//**
 * @param a_moniker             [in]    The moniker for *this*.
 * @param a_type                [in]    The *formType* the class represents.
 * @param a_label               [in]    The label for *this*.
 * @param a_dimension           [in]    The dimension of the function.
 * @param a_index               [in]    Currently not used.
 * @param a_outerDomainValue    [in]    If embedded in a higher dimensional function, the value of the domain of the next higher dimension.
 ***********************************************************************************************************/

FunctionForm::FunctionForm( std::string const &a_moniker, formType a_type, std::string const &a_label, int a_dimension, int a_index, double a_outerDomainValue ) :
        Form( a_type, a_label ),
        m_dimension( a_dimension ),
        m_domainUnit( "" ),
        m_rangeUnit( "" ),
        m_interpolation( ptwXY_interpolationLinLin ),
        m_index( a_index ),
        m_outerDomainValue( a_outerDomainValue ) {

    moniker( a_moniker );

}

/* *********************************************************************************************************//**
 * @param a_type                [in]    The *formType* the class represents.
 * @param a_dimension           [in]    The dimension of the function.
 * @param a_domainUnit          [in]    The domain unit. If dimension is greater than 1, the unit for the outer most independent axis.
 * @param a_rangeUnit           [in]    The unit of the function.
 * @param a_interpolation       [in]    The interpolation along the outer most independent axis and the dependent axis.
 * @param a_index               [in]    Currently not used.
 * @param a_outerDomainValue    [in]    If embedded in a higher dimensional function, the value of the domain of the next higher dimension.
 ***********************************************************************************************************/

FunctionForm::FunctionForm( formType a_type, int a_dimension, std::string const &a_domainUnit, std::string const &a_rangeUnit,
            ptwXY_interpolation a_interpolation, int a_index, double a_outerDomainValue ) :
        Form( a_type ),
        m_dimension( a_dimension ),
        m_domainUnit( a_domainUnit ),
        m_rangeUnit( a_rangeUnit ),
        m_interpolation( a_interpolation ),
        m_index( a_index ),
        m_outerDomainValue( a_outerDomainValue ) {

};

/* *********************************************************************************************************//**
 * @param a_construction    [in]    Used to pass user options to the constructor.
 * @param a_node            [in]    The **pugi::xml_node** to be parsed and used to construct the FunctionForm.
 * @param a_type            [in]    The *formType* the class represents.
 * @param a_dimension       [in]    The dimension of the function.
 * @param a_suite           [in]    The parent GIDI::Suite.
 ***********************************************************************************************************/

FunctionForm::FunctionForm( Construction::Settings const &a_construction, pugi::xml_node const &a_node, formType a_type, int a_dimension, Suite *a_suite ) :
        Form( a_node, a_type, a_suite ),
        m_dimension( a_dimension ),
        m_interpolation( ptwXY_interpolationLinLin ),
        m_index( 0 ), 
        m_outerDomainValue( 0.0 ),
        m_axes( a_node.child( "axes" ), 0 ) {

        m_interpolationString = a_node.attribute( "interpolation" ).value( );
        m_interpolation = ptwXY_stringToInterpolation( m_interpolationString.c_str( ) );
        if( m_interpolation != ptwXY_interpolationOther ) m_interpolationString = ptwXY_interpolationToString( m_interpolation );

        if( strcmp( a_node.attribute( "index" ).value( ), "" ) != 0 ) m_index = a_node.attribute( "index" ).as_int( );
        if( strcmp( a_node.attribute( "outerDomainValue" ).value( ), "" ) != 0 ) m_outerDomainValue = a_node.attribute( "outerDomainValue" ).as_double( );

        if( m_axes.size( ) > 0 ) {
            m_domainUnit = m_axes[0]->unit( );
            m_rangeUnit = m_axes[m_axes.dimension( )]->unit( );
        }

}

/* *********************************************************************************************************//**
 * @param a_form            [in]    The FunctionForm to copy.
 ***********************************************************************************************************/

FunctionForm::FunctionForm( FunctionForm const &a_form ) :
        Form( a_form ),
        m_dimension( a_form.dimension( ) ),
        m_domainUnit( a_form.domainUnit( ) ),
        m_rangeUnit( a_form.rangeUnit( ) ),
        m_interpolation( a_form.interpolation( ) ),
        m_index( a_form.index( ) ),
        m_outerDomainValue( a_form.outerDomainValue( ) ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

FunctionForm::~FunctionForm( ) {

}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 * @param       a_embedded          [in]        If *true*, *this* function is embedded in a higher dimensional function.
 * @param       a_inRegions         [in]        If *true*, *this* is in a Regions container.
 ***********************************************************************************************************/

void FunctionForm::toXMLList_func( WriteInfo &a_writeInfo, std::string const &a_indent, bool a_embedded, bool a_inRegions ) const {

    std::cout << "Node '" << moniker( ) << "' needs toXMLList methods." << std::endl;
}

/*! \class Function1dForm
 * Base class inherited by other GIDI 1d function classes.
 */

/* *********************************************************************************************************//**
 * @param a_type                [in]    The *formType* the class represents.
 * @param a_domainUnit          [in]    The domain unit. If dimension is greater than 1, the unit for the outer most independent axis.
 * @param a_rangeUnit           [in]    The unit of the function.
 * @param a_interpolation       [in]    The interpolation along the outer most independent axis and the dependent axis.
 * @param a_index               [in]    Currently not used.
 * @param a_outerDomainValue    [in]    If embedded in a higher dimensional function, the value of the domain of the next higher dimension.
 ***********************************************************************************************************/

Function1dForm::Function1dForm( formType a_type, std::string const &a_domainUnit, std::string const &a_rangeUnit,
                ptwXY_interpolation a_interpolation, int a_index, double a_outerDomainValue ) :
        FunctionForm( a_type, 1, a_domainUnit, a_rangeUnit, a_interpolation, a_index, a_outerDomainValue ) {

}

/* *********************************************************************************************************//**
 * @param a_construction    [in]    Used to pass user options to the constructor.
 * @param a_node            [in]    The **pugi::xml_node** to be parsed and used to construct the FunctionForm.
 * @param a_type            [in]    The *formType* the class represents.
 * @param a_suite           [in]    The parent GIDI::Suite.
 ***********************************************************************************************************/

Function1dForm::Function1dForm( Construction::Settings const &a_construction, pugi::xml_node const &a_node, formType a_type, Suite *a_suite ) :
        FunctionForm( a_construction, a_node, a_type, 1, a_suite ) {

}

/* *********************************************************************************************************//**
 * @param a_form            [in]    The Function1dForm to copy.
 ***********************************************************************************************************/

Function1dForm::Function1dForm( Function1dForm const &a_form ) :
        FunctionForm( a_form ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

Function1dForm::~Function1dForm( ) {

}

/* *********************************************************************************************************//**
 * @param a_moniker             [in]    The **GNDS** node name for the 2d function.
 * @param a_type                [in]    The *formType* the class represents.
 * @param a_label               [in]    The label for *this*.
 * @param a_index               [in]    The GNDS **index** value for *this*.
 * @param a_outerDomainValue    [in]    The GNDS **outerDomainValue** value for *this*.
 ***********************************************************************************************************/

Function2dForm::Function2dForm( std::string const &a_moniker, formType a_type, std::string const &a_label, int a_index, double a_outerDomainValue ) :
        FunctionForm( a_moniker, a_type, a_label, 2, a_index, a_outerDomainValue ) {

}

/* *********************************************************************************************************//**
 * @param a_construction    [in]    Used to pass user options to the constructor.
 * @param a_node            [in]    The **pugi::xml_node** to be parsed and used to construct the FunctionForm.
 * @param a_type            [in]    The *formType* the class represents.
 * @param a_suite           [in]    The parent GIDI::Suite.
 ***********************************************************************************************************/

Function2dForm::Function2dForm( Construction::Settings const &a_construction, pugi::xml_node const &a_node, formType a_type, Suite *a_suite ) :
        FunctionForm( a_construction, a_node, a_type, 2, a_suite ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

Function2dForm::~Function2dForm( ) {

}

/* *********************************************************************************************************//**
 * @param a_construction    [in]    Used to pass user options to the constructor.
 * @param a_node            [in]    The **pugi::xml_node** to be parsed and used to construct the FunctionForm.
 * @param a_type            [in]    The *formType* the class represents.
 * @param a_suite           [in]    The parent GIDI::Suite.
 ***********************************************************************************************************/

Function3dForm::Function3dForm( Construction::Settings const &a_construction, pugi::xml_node const &a_node, formType a_type, Suite *a_suite ) :
        FunctionForm( a_construction, a_node, a_type, 3, a_suite ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

Function3dForm::~Function3dForm( ) {

}

}
