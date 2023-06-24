#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 09:24:02 2022

@author: ac
"""

# BASE VERBATIM SANS ATTRITION 12 MOIS #
comments_attr_sans = comments_enquete_client.loc[comments_enquete_client['ATTR_ALL']=='SANS']

# BASE VERBATIM SANS ATTRITION CONCURRENCE 12 MOIS #
comments_attr_conc_sans = comments_enquete_client.loc[comments_enquete_client['ATTR_CONC']=='SANS']

# BASE VERBATIM AVEC ATTRITION PARTIELLE 12 MOIS #
comments_attr_conc_part = comments_enquete_client.loc[comments_enquete_client['ATTR_CONC']=='PART']

# BASE VERBATIM AVEC ATTRITION TOTALE 12 MOIS #
comments_attr_conc_tot = comments_enquete_client.loc[comments_enquete_client['ATTR_CONC']=='TOT']

