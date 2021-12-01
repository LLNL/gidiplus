/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include "HAPI.hpp"

namespace HAPI {

/*
=========================================================
 *
 * @return
 */
Node::Node() :
        m_node(NULL) {

}
/*
=========================================================
 *
 * @param a_node
 * @return
 */
Node::Node( Node_internal *a_node ) :
        m_node(a_node) {

}
/*
=========================================================
*/
Node::~Node( ) {

    delete m_node;

}
/*
============================================================
===================== get child element ====================
============================================================
 *
 * @return
 */
Node Node::child(char const *a_name) const {

    if (NULL == m_node)
        return Node();
    return Node( m_node->child( a_name ) );

}
/*
=========================================================
*/
Node Node::first_child() const {

    if (NULL == m_node)
        return Node();
    return Node( m_node->first_child( ) );

}
/*
============================================================
===================== get sibling element ==================
============================================================
 *
 * @return
 */
Node Node::next_sibling() const {

    if (NULL == m_node)
        return Node();
    Node_internal *sibling = m_node->next_sibling( );
    delete m_node;
    return Node( sibling );

}
/*
============================================================
============ update self to point to next sibling ==========
============================================================
 *
 * @return
 */
void Node::to_next_sibling() const {

    m_node->to_next_sibling( );

}
/*
============================================================
===================== assignment operator ==================
============================================================
 */
Node& Node::operator=(const Node &other) {

    if (NULL != other.m_node)
        this->m_node = other.m_node->copy();
    return *this;

}
/*
============================================================
===================== get tag name =========================
============================================================
 *
 * @return
 */
std::string Node::name() const {

    if (NULL == m_node)
        return std::string("");
    return m_node->name();

}
/*
============================================================
================== test for empty node =====================
============================================================
 *
 * @return
 */
bool Node::empty() const {

    if (NULL == m_node)
        return true;
    return m_node->empty();

}
/*
============================================================
======================= text data ==========================
============================================================
 *
 * @return
 */
Text Node::text() const {

    if (NULL == m_node)
        return Text();
    return m_node->text();

}
/*
============================================================
===================== numeric data =========================
============================================================
 *
 * @return
 */
Data Node::data() const {

    if (NULL == m_node)
        return Data();
    return Data( m_node->data() );

}

}
