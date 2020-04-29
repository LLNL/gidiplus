/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include "MCGIDI.hpp"
#include <math.h>

namespace MCGIDI {

/* *********************************************************************************************************//**
 * This function determines the probability that a product of index *secondary_particle_pops_id* is emitted
 * at angle mu_p3_lab in the lab frame.
 ***********************************************************************************************************/

HOST_DEVICE void angleBiasing( int secondary_particle_pops_id,
                               double wgt_pre_bias,
                               double mu_p3_lab,
                               double KE_p1_lab,
                               const Protare *gidi_interaction_data,
                               const Reaction *gidi_reaction,
                               Sampling::ClientCodeRNGData rng_data,
                               double &wgt_p3,
                               double &KE_p3_lab,
                               double &speed ) {

    double c2 = MCGIDI_speedOfLight_cm_sec * MCGIDI_speedOfLight_cm_sec;
    const double m_p1 = gidi_reaction->projectileMass();
    const double m_p2 = gidi_reaction->targetMass();
    double speed_p1_lab = sqrt(KE_p1_lab * 2.0 / m_p1 * c2);
    double speed_m_lab = speed_p1_lab * m_p1 / (m_p1 + m_p2); // cm/s
    double speed_m_lab2 = speed_m_lab * speed_m_lab;
    // see comment about excitation energies below
    //const double E_ex_p1 = gidi_interaction_data->projectileExcitationEnergy();
    //const double E_ex_p2 = gidi_interaction_data->targetExcitationEnergy();
    Vector<Product *> const &products = gidi_reaction->outputChannel( ).products( );

    // loop over all products from reaction that actually occurred
    for ( MCGIDI_VectorSizeType product_index = 0;
         product_index < products.size();
         product_index++)
    {
        Product *product = products[product_index];
        product->angleBiasingPDFValue(0.0);
        if (product->index() != secondary_particle_pops_id)
        {
            continue;
        }

        switch (product->distribution()->type())
        {
        case Distributions::Type::none:
            {
                THROW( "MCGIDI::angleBiasing: diagnostic particle cannot have a distribution Type MCGIDI::Distributions::Type::none" );
                break;
            }

        case Distributions::Type::unspecified:
            {
                THROW( "MC_Launch_Diagnostic_Particles: diagnostic particle cannot have a distribution Type MCGIDI::Distributions::Type::unspecified" );
                break;
            }

        case Distributions::Type::angularTwoBody:
            {
                // do non-relativistic two-body kinematics
                // ---------------------------------------

                // ensure p3 is always the current product_index

                Distributions::AngularTwoBody const *distribution = static_cast<Distributions::AngularTwoBody const *>( product->distribution( ) );
                Product *p3 = nullptr;
                double m_p4 = 0.0;

                if (product_index == 0)
                {
                    p3 = products[0];
                    m_p4 = distribution->residualMass( );                       // Includes nuclear excitation energy.
                }
                else if (product_index == 1)
                {
                    p3 = products[1];
                    m_p4 = distribution->productMass( );                        // Includes nuclear excitation energy.
                }

                double m_p3 = p3->mass( );                                      // Includes nuclear excitation energy.

                // product class masses (are supposed to) include excitation energy per MCGIDI_product.cpp comments
#if 0
                double E_ex_p3 = p3->excitationEnergy();
                double E_ex_p4 = p4->excitationEnergy();

                // different options for COM kinetic energy of secondary particle depending on what is actually in the data
                double KE_p3_com = (KE_p1_com + KE_p2_com           \
                                    + m_p1 + m_p2 - m_p3 - m_p4     \
                                    + E_ex_p1 + E_ex_p2 - E_ex_p3 - E_ex_p4) \
                    / (1.0 + (m_p3 + E_ex_p3) / (m_p4 + E_ex_p4));

                double KE_p3_com = (KE_p1_com + KE_p2_com           \
                                    + m_p1 + m_p2 - m_p3 - m_p4)    \
                    / (1.0 + m_p3 / m_p4);

                double KE_p3_com = (KE_p1_com + KE_p2_com           \
                                    - m_p2 / (m_p1 + m_p2) * gidi_reaction->crossSectionThreshold()) \
                    / (1.0 + m_p3 / m_p4);
#endif

                double Q_val = -m_p2 / (m_p1 + m_p2) * gidi_reaction->crossSectionThreshold();
                m_p4 += -Q_val; // need to actually have mass include excitation
                double mu_p3_lab2 = mu_p3_lab * mu_p3_lab;
                double pdf_val_1 = 0.0;
                double pdf_val_2 = 0.0;
                p3->angleBiasingPDFValue(0.0);
                double mu_p3_com_1 = 0.0;
                double mu_p3_com_2 = 0.0;
                double dmu_p3_com_1 = 0.0;
                double dmu_p3_com_2 = 0.0;
                double speed_p3_lab_1 = 0.0;
                double speed_p3_lab_2 = 0.0;

                // alternate expression for secondary particle lab kinetic energy from Foderaro
                //   (direct calculation of energy from mu_p3_lab --- instead of going thru COM first)
#if 0
                double KE_p3_lab_Foderaro = KE_p1_lab   \
                    * (m_p1 / (m_p1 + m_p2) * mu_p3_lab + sqrt((m_p2 - m_p1)/(m_p1 + m_p2) \
                                                               + (m_p1/(m_p1+m_p2))*(m_p1/(m_p1+m_p2))*mu_p3_lab2-m_p2/(m_p1+m_p2)*m_p2 \
                                                               / (m_p1 + m_p2) * gidi_reaction->crossSectionThreshold()/KE_p1_lab)) \
                    * (m_p1 / (m_p1 + m_p2) * mu_p3_lab + sqrt((m_p2 - m_p1)/(m_p1 + m_p2) \
                                                               + (m_p1/(m_p1+m_p2))*(m_p1/(m_p1+m_p2))*mu_p3_lab2-m_p2/(m_p1+m_p2)*m_p2 \
                                                               / (m_p1 + m_p2) * gidi_reaction->crossSectionThreshold()/KE_p1_lab));
#endif

                double speed_p2_com = speed_m_lab;
                double speed_p1_com = speed_m_lab * m_p2 / m_p1; // cm/s
                double KE_p1_com = 0.5 * m_p1 * speed_p1_com * speed_p1_com / c2;
                double KE_p2_com = 0.5 * m_p2 * speed_p2_com * speed_p2_com / c2;
                double KE_p3_com = (KE_p1_com + KE_p2_com + Q_val) / (1.0 + m_p3 / m_p4);
                double speed_p3_com = sqrt(KE_p3_com * 2.0 / m_p3 * c2);
                double speed_ratio = speed_m_lab / speed_p3_com;
                double speed_ratio2 = speed_ratio * speed_ratio;

                if (mu_p3_lab2 * (1.0 + speed_ratio2 * (mu_p3_lab2 - 1.0)) < 0.0) { break; }

                double lead_term = speed_ratio * (mu_p3_lab2 - 1.0);
                double radical_term = sqrt(mu_p3_lab2 * (1.0 + speed_ratio2 * (mu_p3_lab2 - 1.0)));
                mu_p3_com_1 = lead_term - radical_term;
                mu_p3_com_2 = lead_term + radical_term;
                lead_term = 2.0 * speed_ratio * mu_p3_lab;
                radical_term = (mu_p3_lab * (1.0 + speed_ratio2 * (2.0 * mu_p3_lab2 - 1.0))) / sqrt(mu_p3_lab2 * (1.0 + speed_ratio2 * (mu_p3_lab2 - 1.0)));
                dmu_p3_com_1 = lead_term - radical_term;
                dmu_p3_com_2 = lead_term + radical_term;

                // "Usual" procedure: work within the COM, seems to produce (spectral) answers
                // slightly inconsistent with analog tallies (is this due to numerics,
                // the way the analog kinematics are done in MCGIDI, etc.?)
#if 0
                double speed_p3_com2 = speed_p3_com * speed_p3_com;
                if (fabs(mu_p3_com_1) <= 1.0)
                {
                    speed_p3_lab_1 = speed_m_lab * mu_p3_lab + sqrt(speed_m_lab2 * (mu_p3_lab2 - 1.0) + speed_p3_com2);

                    // alternate expression using Foderaro's secondary particle lab energy
                    //speed_p3_lab_1 = sqrt(KE_p3_lab_Foderaro * 2.0 / m_p3 * c2);
                                          
                    // alternate expression
                    //speed_p3_lab_1 = sqrt((speed_m_lab + speed_p3_com * mu_p3_com_1)*(speed_m_lab + speed_p3_com * mu_p3_com_1)
                    //+ speed_p3_com2 * sin(acos(mu_p3_com_1)) * sin(acos(mu_p3_com_1)));
                    if (speed_p3_lab_1 > 0.0 && speed_p3_lab_1 < speed_p1_lab)
                    {
                        p3->evaluate_angular_pdf(KE_p1_lab, mu_p3_com_1, pdf_val_1);
                        p3->angleBiasingPDFValue( p3->angleBiasingPDFValue() + fabs(dmu_p3_com_1) * pdf_val_1 );
                    }
                }
                if (fabs(mu_p3_com_2) <= 1.0)
                {
                    speed_p3_lab_2 = speed_m_lab * mu_p3_lab + sqrt(speed_m_lab2 * (mu_p3_lab2 - 1.0) + speed_p3_com2);
                    //speed_p3_lab_2 = sqrt(KE_p3_lab_Foderaro * 2.0 / m_p3 * c2);
                    //speed_p3_lab_2 = sqrt((speed_m_lab + speed_p3_com * mu_p3_com_2)*(speed_m_lab + speed_p3_com * mu_p3_com_2)
                    //+ speed_p3_com2 * sin(acos(mu_p3_com_2)) * sin(acos(mu_p3_com_2)));
                    if (speed_p3_lab_2 > 0.0 && speed_p3_lab_2 < speed_p1_lab)
                    {
                        p3->evaluate_angular_pdf(KE_p1_lab, mu_p3_com_2, pdf_val_2);
                        p3->angleBiasingPDFValue( p3->angleBiasingPDFValue() + fabs(dmu_p3_com_2) * pdf_val_2 );
                    }
                }
#endif

                // Yip/Foderaro-inspired procedure: work with lab energies, seems to produce answers
                // consistent with analog tallies (is this due to numerics, the way the analog kinematics are done, etc.?)
                double E_p3_lab_1;
                double E_p3_lab_2;
                double radical_tmp = (m_p3 + m_p4) * (-KE_p1_lab * m_p1 + m_p4 * (Q_val + KE_p1_lab)) + KE_p1_lab * m_p1 * m_p3 * mu_p3_lab2;

                if (radical_tmp < 0.0) { break; }

                double sqrt_E_p3_lab_1;
                double sqrt_E_p3_lab_2;
                sqrt_E_p3_lab_1 = (sqrt(KE_p1_lab * m_p1 * m_p3) * mu_p3_lab - sqrt(radical_tmp)) / (m_p3 + m_p4);
                sqrt_E_p3_lab_2 = (sqrt(KE_p1_lab * m_p1 * m_p3) * mu_p3_lab + sqrt(radical_tmp)) / (m_p3 + m_p4);
                if (sqrt_E_p3_lab_1 > 0.0)
                {
                    E_p3_lab_1 = sqrt_E_p3_lab_1 * sqrt_E_p3_lab_1;
                    speed_p3_lab_1 = sqrt(E_p3_lab_1 * 2.0 / m_p3 * c2);
                    double theta_p3_com;
                    double theta_p3_lab;
                    theta_p3_lab = acos(mu_p3_lab);
                    theta_p3_com = atan2(speed_p3_lab_1 * sin(theta_p3_lab), speed_p3_lab_1 * mu_p3_lab - speed_m_lab);
                    mu_p3_com_1 = cos(theta_p3_com);
                    p3->evaluate_angular_pdf(KE_p1_lab, mu_p3_com_1, pdf_val_1);
                    p3->angleBiasingPDFValue( p3->angleBiasingPDFValue() + fabs(dmu_p3_com_1) * pdf_val_1 );
                }
                if (sqrt_E_p3_lab_2 > 0.0)
                {
                    E_p3_lab_2 = sqrt_E_p3_lab_2 * sqrt_E_p3_lab_2;
                    speed_p3_lab_2 = sqrt(E_p3_lab_2 * 2.0 / m_p3 * c2);
                    double theta_p3_com;
                    double theta_p3_lab;
                    theta_p3_lab = acos(mu_p3_lab);
                    theta_p3_com = atan2(speed_p3_lab_2 * sin(theta_p3_lab), speed_p3_lab_2 * mu_p3_lab - speed_m_lab);
                    mu_p3_com_2 = cos(theta_p3_com);
                    p3->evaluate_angular_pdf(KE_p1_lab, mu_p3_com_2, pdf_val_2);
                    p3->angleBiasingPDFValue( p3->angleBiasingPDFValue() + fabs(dmu_p3_com_2) * pdf_val_2 );
                }

                // select speed to assign secondary particle based on the relative probability of exiting at that speed
                // (as opposed to the other possible speed) since there are two possible speeds due to there being
                // two possible outgoing cosines
                double random_num = rng_data.dRng();
//printf("%s:%d DEBUG random_num=%e\n",__FILE__,__LINE__,random_num); fflush(stdout);
                double speed_p3_lab;
//printf("%s:%d DEBUG pdf_val_1=%e\n",__FILE__,__LINE__,pdf_val_1); fflush(stdout);
//printf("%s:%d DEBUG dmu_p3_com_1=%e\n",__FILE__,__LINE__,dmu_p3_com_1); fflush(stdout);
//double temp = p3->angleBiasingPDFValue();
//printf("%s:%d DEBUG p3->angleBiasingPDFValue=%e\n",__FILE__,__LINE__,temp); fflush(stdout);
                if ( (random_num * p3->angleBiasingPDFValue()) < (pdf_val_1 * fabs(dmu_p3_com_1)) )
                {
                    speed_p3_lab = speed_p3_lab_1;
                }
                else
                {
                    speed_p3_lab = speed_p3_lab_2;
                }

                if (speed_p3_lab < 0.0 || speed_p3_lab > speed_p1_lab)
                {
                    p3->angleBiasingPDFValue( 0.0 );
                }

                p3->angleBiasingEnergy( 0.5 * m_p3 * speed_p3_lab * speed_p3_lab / c2 );
                p3->angleBiasingSpeed( speed_p3_lab );

                break;
            }

        case Distributions::Type::KalbachMann:
            {
                Sampling::Input sampling_input(true, Sampling::Upscatter::Model::none);
                product->distribution()->sample(KE_p1_lab, sampling_input, rng_data.rng( ), rng_data.rngState( ) );
                double KE_p3_com = sampling_input.m_energyOut1;
                double m_p3 = product->mass();
                double speed_p3_com = sqrt(KE_p3_com * 2.0 / m_p3 * c2);
                double speed_p3_com2 = speed_p3_com * speed_p3_com;
                double speed_ratio = speed_m_lab / speed_p3_com;
                double speed_ratio2 = speed_ratio * speed_ratio;
                double mu_p3_lab2 = mu_p3_lab * mu_p3_lab;
                double pdf_val_1 = 0.0;
                double pdf_val_2 = 0.0;
                product->angleBiasingPDFValue(0.0);
                double dmu_p3_com_1 = 0.0;
                double dmu_p3_com_2 = 0.0;
                double speed_p3_lab_1 = 0.0;
                double speed_p3_lab_2 = 0.0;
                double lead_term = 0.0;
                double radical_term = 0.0;
                double mu_p3_com_1 = 0.0;
                double mu_p3_com_2 = 0.0;

                if (mu_p3_lab2 * (1.0 + speed_ratio2 * (mu_p3_lab2 - 1.0)) < 0.0) { break; }

                lead_term = speed_ratio * (mu_p3_lab2 - 1.0);
                radical_term = sqrt(mu_p3_lab2 * (1.0 + speed_ratio2 * (mu_p3_lab2 - 1.0)));
                mu_p3_com_1 = lead_term - radical_term;
                mu_p3_com_2 = lead_term + radical_term;
                lead_term = 2.0 * speed_ratio * mu_p3_lab;
                radical_term = (1.0 + speed_ratio2 * (2.0 * mu_p3_lab2 - 1.0)) / sqrt(1.0 + speed_ratio2 * (mu_p3_lab2 - 1.0));
                dmu_p3_com_1 = lead_term - radical_term;
                dmu_p3_com_2 = lead_term + radical_term;

                if (fabs(mu_p3_com_1) <= 1.0)
                {
                    speed_p3_lab_1 = speed_m_lab * mu_p3_lab + sqrt(speed_m_lab2 * (mu_p3_lab2 - 1.0) + speed_p3_com2);
                    product->evaluate_angular_pdf(KE_p1_lab, KE_p3_com, mu_p3_com_1, pdf_val_1);
                    product->angleBiasingPDFValue( product->angleBiasingPDFValue() + fabs(dmu_p3_com_1) * pdf_val_1);
                }
                if (fabs(mu_p3_com_2) <= 1.0)
                {
                    speed_p3_lab_2 = speed_m_lab * mu_p3_lab + sqrt(speed_m_lab2 * (mu_p3_lab2 - 1.0) + speed_p3_com2);
                    product->evaluate_angular_pdf(KE_p1_lab, KE_p3_com, mu_p3_com_2, pdf_val_2);
                    product->angleBiasingPDFValue( product->angleBiasingPDFValue() + fabs(dmu_p3_com_2) * pdf_val_2 );
                }

                double random_num = rng_data.dRng();
                double speed_p3_lab;
                if ( (random_num * product->angleBiasingPDFValue()) < (pdf_val_1 * fabs(dmu_p3_com_1)) )
                {
                    speed_p3_lab = speed_p3_lab_1;
                }
                else
                {
                    speed_p3_lab = speed_p3_lab_2;
                }

                product->angleBiasingEnergy( 0.5 * m_p3 * speed_p3_lab * speed_p3_lab / c2 );
                product->angleBiasingSpeed( speed_p3_lab );

                break;
            }

        case Distributions::Type::uncorrelated:
            {
                if (product->distribution()->productFrame() == GIDI::Frame::lab)
                {
                    double pdf_val;
                    product->evaluate_angular_pdf(KE_p1_lab, mu_p3_lab, pdf_val);
                    product->angleBiasingPDFValue( pdf_val );
                    Sampling::Input sampling_input(true, Sampling::Upscatter::Model::none);
                    product->distribution()->sample(KE_p1_lab, sampling_input, rng_data.rng(), rng_data.rngState() );
                    product->angleBiasingEnergy( sampling_input.m_energyOut1 );
                    double m_p3 = product->mass();
                    double speed_p3_lab = sqrt(product->angleBiasingEnergy() * 2.0 / m_p3 * c2);
                    product->angleBiasingSpeed( speed_p3_lab );
                }
                else // COM frame
                {
                    Sampling::Input sampling_input(true, Sampling::Upscatter::Model::none);
                    product->distribution()->sample(KE_p1_lab, sampling_input, rng_data.rng(), rng_data.rngState() );
                    double KE_p3_com = sampling_input.m_energyOut1;
                    double m_p3 = product->mass();
                    double speed_p3_com = sqrt(KE_p3_com * 2.0 / m_p3 * c2);
                    double speed_p3_com2 = speed_p3_com * speed_p3_com;
                    double speed_ratio = speed_m_lab / speed_p3_com;
                    double speed_ratio2 = speed_ratio * speed_ratio;
                    double mu_p3_lab2 = mu_p3_lab * mu_p3_lab;
                    double pdf_val_1 = 0.0;
                    double pdf_val_2 = 0.0;
                    product->angleBiasingPDFValue( 0.0 );
                    double dmu_p3_com_1 = 0.0;
                    double dmu_p3_com_2 = 0.0;
                    double speed_p3_lab_1 = 0.0;
                    double speed_p3_lab_2 = 0.0;
                    if (mu_p3_lab2 * (1.0 + speed_ratio2 * (mu_p3_lab2 - 1.0)) < 0.0)
                    {
                        break;
                    }
                    else
                    {
                        double lead_term = speed_ratio * (mu_p3_lab2 - 1.0);
                        double radical_term = sqrt(mu_p3_lab2 * (1.0 + speed_ratio2 * (mu_p3_lab2 - 1.0)));
                        double mu_p3_com_1 = lead_term - radical_term;
                        double mu_p3_com_2 = lead_term + radical_term;
                        lead_term = 2.0 * speed_ratio * mu_p3_lab;
                        radical_term = (1.0 + speed_ratio2 * (2.0 * mu_p3_lab2 - 1.0)) / sqrt(1.0 + speed_ratio2 * (mu_p3_lab2 - 1.0));
                        dmu_p3_com_1 = lead_term - radical_term;
                        dmu_p3_com_2 = lead_term + radical_term;
                        if (fabs(mu_p3_com_1) <= 1.0)
                        {
                            speed_p3_lab_1 = speed_m_lab * mu_p3_lab + sqrt(speed_m_lab2 * (mu_p3_lab2 - 1.0) + speed_p3_com2);
                            product->evaluate_angular_pdf(KE_p1_lab, mu_p3_com_1, pdf_val_1);
                            product->angleBiasingPDFValue( product->angleBiasingPDFValue() + fabs(dmu_p3_com_1) * pdf_val_1);
                        }
                        if (fabs(mu_p3_com_2) <= 1.0)
                        {
                            speed_p3_lab_2 = speed_m_lab * mu_p3_lab + sqrt(speed_m_lab2 * (mu_p3_lab2 - 1.0) + speed_p3_com2);
                            product->evaluate_angular_pdf(KE_p1_lab, mu_p3_com_2, pdf_val_2);
                            product->angleBiasingPDFValue( product->angleBiasingPDFValue() + fabs(dmu_p3_com_2) * pdf_val_2);
                        }
                    }

                    double random_num = rng_data.dRng();
                    double speed_p3_lab;
                    if ( (random_num *product->angleBiasingPDFValue()) < (pdf_val_1 * fabs(dmu_p3_com_1)) )
                    {
                        speed_p3_lab = speed_p3_lab_1;
                    }
                    else
                    {
                        speed_p3_lab = speed_p3_lab_2;
                    }

                    product->angleBiasingEnergy( 0.5 * m_p3 * speed_p3_lab * speed_p3_lab / c2 );
                    product->angleBiasingSpeed( speed_p3_lab );
                }

                break;
            }

        case Distributions::Type::energyAngularMC:
            {
                if (product->distribution()->productFrame() == GIDI::Frame::lab)
                {
                    Sampling::Input sampling_input(true, Sampling::Upscatter::Model::none);
                    product->distribution()->sample(KE_p1_lab, sampling_input, rng_data.rng(), rng_data.rngState() );
                    product->angleBiasingEnergy( sampling_input.m_energyOut1 );
                    double pdf_val;
                    product->evaluate_angular_pdf(KE_p1_lab, product->angleBiasingEnergy(), mu_p3_lab, pdf_val);
                    product->angleBiasingPDFValue( pdf_val );
                    double m_p3 = product->mass();
                    double speed_p3_lab = sqrt(product->angleBiasingEnergy() * 2.0 / m_p3 * c2);
                    product->angleBiasingSpeed( speed_p3_lab );
                }
                else // COM frame
                {
                    Sampling::Input sampling_input(true, Sampling::Upscatter::Model::none);
                    product->distribution()->sample(KE_p1_lab, sampling_input, rng_data.rng(), rng_data.rngState() );
                    double KE_p3_com = sampling_input.m_energyOut1;
                    double m_p3 = product->mass();
                    double speed_p3_com = sqrt(KE_p3_com * 2.0 / m_p3 * c2);
                    double speed_p3_com2 = speed_p3_com * speed_p3_com;
                    double speed_ratio = speed_m_lab / speed_p3_com;
                    double speed_ratio2 = speed_ratio * speed_ratio;
                    double mu_p3_lab2 = mu_p3_lab * mu_p3_lab;
                    double pdf_val_1 = 0.0;
                    double pdf_val_2 = 0.0;
                    product->angleBiasingPDFValue( 0.0 );
                    double dmu_p3_com_1 = 0.0;
                    double dmu_p3_com_2 = 0.0;
                    double speed_p3_lab_1 = 0.0;
                    double speed_p3_lab_2 = 0.0;

                    if (mu_p3_lab2 * (1.0 + speed_ratio2 * (mu_p3_lab2 - 1.0)) < 0.0)
                    {
                        break;
                    }
                    else
                    {
                        double lead_term = speed_ratio * (mu_p3_lab2 - 1.0);
                        double radical_term = sqrt(mu_p3_lab2 * (1.0 + speed_ratio2 * (mu_p3_lab2 - 1.0)));
                        double mu_p3_com_1 = lead_term - radical_term;
                        double mu_p3_com_2 = lead_term + radical_term;
                        lead_term = 2.0 * speed_ratio * mu_p3_lab;
                        radical_term = (1.0 + speed_ratio2 * (2.0 * mu_p3_lab2 - 1.0)) / sqrt(1.0 + speed_ratio2 * (mu_p3_lab2 - 1.0));
                        dmu_p3_com_1 = lead_term - radical_term;
                        dmu_p3_com_2 = lead_term + radical_term;
                        if (fabs(mu_p3_com_1) <= 1.0)
                        {
                            speed_p3_lab_1 = speed_m_lab * mu_p3_lab + sqrt(speed_m_lab2 * (mu_p3_lab2 - 1.0) + speed_p3_com2);
                            product->evaluate_angular_pdf(KE_p1_lab, KE_p3_com, mu_p3_com_1, pdf_val_1);
                            product->angleBiasingPDFValue( product->angleBiasingPDFValue() + fabs(dmu_p3_com_1) * pdf_val_1 );
                        }
                        if (fabs(mu_p3_com_2) <= 1.0)
                        {
                            speed_p3_lab_2 = speed_m_lab * mu_p3_lab + sqrt(speed_m_lab2 * (mu_p3_lab2 - 1.0) + speed_p3_com2);
                            product->evaluate_angular_pdf(KE_p1_lab, KE_p3_com, mu_p3_com_2, pdf_val_2);
                            product->angleBiasingPDFValue( product->angleBiasingPDFValue() + fabs(dmu_p3_com_2) * pdf_val_2 );
                        }
                    }

                    double random_num = rng_data.dRng();
                    double speed_p3_lab;
                    if ( (random_num * product->angleBiasingPDFValue()) < (pdf_val_1 * fabs(dmu_p3_com_1)) )
                    {
                        speed_p3_lab = speed_p3_lab_1;
                    }
                    else
                    {
                        speed_p3_lab = speed_p3_lab_2;
                    }

                    product->angleBiasingEnergy( 0.5 * m_p3 * speed_p3_lab * speed_p3_lab / c2 );
                    product->angleBiasingSpeed( speed_p3_lab );
                }

                break;
            }

        case Distributions::Type::angularEnergyMC:
            {
                if (product->distribution()->productFrame() == GIDI::Frame::lab)
                {
                    double pdf_val;
                    product->evaluate_angular_pdf(KE_p1_lab, mu_p3_lab, pdf_val);
                    product->angleBiasingPDFValue( pdf_val );
                    double KE_p3_lab_loc;
                    double random_num = rng_data.dRng();
                    Sampling::Input sampling_input(true, Sampling::Upscatter::Model::none);
                    product->distribution()->sample_pdf( KE_p1_lab, mu_p3_lab, KE_p3_lab_loc, random_num, sampling_input, rng_data.rng( ), rng_data.rngState( ) );
                    product->angleBiasingEnergy( KE_p3_lab_loc );
                    double m_p3 = product->mass();
                    double speed_p3_lab = sqrt(product->angleBiasingEnergy() * 2.0 / m_p3 * c2);
                    product->angleBiasingSpeed( speed_p3_lab );
                }
                else // COM frame: evaluations must provide lab frame data (per ENDF-6 format)
                {
                    THROW( "MC_Launch_Diagnostic_Particles: MCGIDI::Distributions::Type::angularEnergyMC data must be in the lab frame" );
                }

                break;
            }

        default:
            {
                THROW( "MC_Launch_Diagnostic_Particles: unrecognized MCGIDI::Distributions Type" );
                break;
            }
        } // product distribution switch case

        double product_multiplicity = product->multiplicity()->evaluate(KE_p1_lab);
        wgt_p3 += 2.0 * wgt_pre_bias * product->angleBiasingPDFValue() * product_multiplicity;

    } // product loop

    // In case multiple product particles of the same species as the tracked
    // diagnostic particle are possible, need to select the lab energy and speed
    // to assign to the single diagnostic particles based on the relative probability
    // of the product particles going out at mu_p3_lab

    double random_num = rng_data.dRng();
    double wgt_cum_sum = 0.0;

    // loop over all products from reaction that actually occurred
    for ( MCGIDI_VectorSizeType product_index = 0;
         product_index < products.size();
         product_index++)
    {
        Product *product = products[product_index];
        if (product->index() != secondary_particle_pops_id)
        {
            continue;
        }

        wgt_cum_sum += 2.0 * wgt_pre_bias * product->angleBiasingPDFValue() * product->multiplicity()->evaluate(KE_p1_lab);

        if ((random_num * wgt_p3) < wgt_cum_sum)
        {
            KE_p3_lab = product->angleBiasingEnergy();
            speed = product->angleBiasingSpeed();
            break;
        }
    } // product loop

} // angleBiasing

} // namespace MCGIDI
