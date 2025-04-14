/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include <stdlib.h>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <sstream>

#include <pugixml.hpp>

#include <LUPI.hpp>

using std::cout;
using std::endl;

int main( LUPI_maybeUnused int argc, LUPI_maybeUnused char **argv ) {

    std::cerr << "    " << "purePugi" << std::endl;

    std::string protareFilename( "../sampleFile.xml" );

    pugi::xml_document doc;
    pugi::xml_attribute attr;

    pugi::xml_parse_result result = doc.load_file( protareFilename.c_str() );
    if (result.status != pugi::status_ok) throw( std::ios_base::failure( result.description() ) );

    pugi::xml_node protare = doc.first_child();

    attr = protare.attribute("projectile");
    cout << attr.name() << ": " << attr.value() << ", ";
    attr = protare.attribute("target");
    cout << attr.name() << ": " << attr.value() << endl;

    pugi::xml_node style = protare.child("styles").first_child();
    cout << "contains styles: " << style.name();
    for (style = style.next_sibling(); style; style = style.next_sibling() )
    {
        cout << ", " << style.name();
    }
    cout << endl;

    pugi::xml_node documentation = protare.child("documentations").child("documentation");
    pugi::xml_text doc_text = documentation.text();
    cout << "documentation: '" << doc_text.get() << "'" << endl;

    pugi::xml_node reaction = protare.child("reactions").first_child();

    cout << "reaction " << reaction.attribute("label").value() << ":" << endl;
    pugi::xml_node xys1d = reaction.child("crossSection").first_child();
    pugi::xml_node values = xys1d.child("values");
    pugi::xml_text text = values.text();
    cout << "  cross section points: ";

    // Manually split string and convert to doubles
    std::istringstream iss(text.get());
    for (std::string s; iss >> s; )
        cout << " " << strtod(s.c_str(), NULL);
    cout << endl;

    pugi::xml_node Qform = reaction.child("outputChannel").child("Q").first_child();
    cout << "  Q-value: " << Qform.name() << " value = " << Qform.attribute("constant").as_int();
    cout << " over range " << Qform.attribute("domainMin").as_double() << " - " << Qform.attribute("domainMax").as_double();
    cout << " " << Qform.child("axes").first_child().attribute("unit").value() << endl;

    pugi::xml_node products = reaction.child("outputChannel").child("products");
    cout << "  products:" << endl;
    for (pugi::xml_node product = products.first_child(); product; product = product.next_sibling())
    {
        cout << "    " << product.attribute("pid").value();
        cout << " multiplicity " << product.child("multiplicity").child("constant1d").attribute("constant").as_int();
        pugi::xml_node distForm = product.child("distribution").first_child();
        cout << " distribution " << distForm.name() << "." << distForm.first_child().name() << endl;
    }

    cout << endl << "Create empty node:";
    pugi::xml_node emptyNode = pugi::xml_node();
    cout << "         name = '" << emptyNode.name() << "', empty = " << emptyNode.empty() << endl;

    cout << "Access non-existent node:";
    pugi::xml_node noSuchNode = protare.child("noSuchNode");
    cout << "  name = '" << noSuchNode.name() << "', empty = " << noSuchNode.empty() << endl;

    cout << "Access non-existent attr:";
    pugi::xml_attribute noSuchAttr = protare.attribute("noSuchAttribute");
    cout << "  name = '" << noSuchAttr.name() << "', value = '" << noSuchAttr.value() << "'" << endl;

}
