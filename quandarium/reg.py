"""This module present functions that extract data from Quantum Chemistry (QC)
Calculations, and related functions. The QC codes with extractors avaliable
are:
- FHI-aims
- VESTA (not yet)
"""

import sys
import pandas as pd
import numpy as np
from quandarium.aux import to_nparray
from quandarium.aux import logcolumns
from quandarium.aux import checkmissingkeys
from quandarium.mols import avradius


def relativise(mainfeature, groupbyfeature):  # Versao Nova
    """It calculate the relative value of a feature for each value of
    groupbyfeature. In other words, the complete dataset will be divided in n
    smaller dataset where the samples divide the same value of groupbyfeature,
    then the relative feature values will be calculated as the value of the
    featuretorelativize minus the smallest value of this feature in the group
    that it belongs to.

    Parameters
    ----------
    mainfeature: np.array, pandas.Series, or list
                 The data of feature which will be compute the relative
                 features.
    groupbyfeature: np.array, pandas.Series, list of None
                    The relativised data will be calculated per group of
                    samples with unique unique values of the groupbyfeature.
                    If None, the relativeised feature will employ the min of
                    the whole data.
    Return
    ------
    relativesed: np.array
                 The reletivised feature values in a np.ndarray.
    """

    print("Initializing analysis: relativising")
    pd_df = pd.DataFrame()
    pd_df['mainfeature'] = to_nparray(mainfeature)
    if groupbyfeature:
        pd_df['groupbyfeature'] = to_nparray(groupbyfeature)
        #print(pd_df, feature, groupbyfeature, to_nparray(feature),to_nparray(groupbyfeature))
        grouped = pd_df.groupby(groupbyfeature)
        relativised = np.zeros(len(pd_df))
        for group in grouped.groups:
            groupedby_df = grouped.get_group(group)
            indexes = groupedby_df.index.tolist()
            newdata = to_nparray(groupedby_df['mainfeature']) - to_nparray(groupedby_df['feature']).min()
            relativised[indexes] = np.array(newdata)
    else:
        relativised = pd_df['mainfeature'].values - pd_df['mainfeature'].values.min()
    return np.array(relativised)


def rec_avradius(positions, useradius=False, davraddii=[], davradius='dav'):  # Versao Nova
    """It calculate the the average radius of some molecule for all molecules
    based in the necleus positions, or including a radius.

    Parameters
    ----------
    positionsfeature: str (optional, default='bag_positions')
                      The name of the fuature (bag type) in pd_df with
                      cartezian positions of the atoms.
    useradius: bool (optional, default=False)
               If True, the radius will be consider to calculate the average radius.
    davraddii: str (optional, default='bag_dav')
                      The name of the fuature in pd_df with atomic radii or dav
                      information (bag of floats).
    davradius: str ['dav','radii'] (optional, default='dav')
               If radii, atomic radius will be the feature davraddiifeature
               values. If dav the values in atomic radius will be half of the
               feature davraddiifeature values.
    Return
    ------
    new_data: np.array.
              The new data in a np.array.
    """
    print("Initializing analysis: rec_avradius")

    positions = to_nparray(positions).tolist()
    davraddii = to_nparray(davraddii).tolist()
    new_data = []
    for index in range(len(positions)):
        positions_i = np.array(positions[index])
        if davradius == 'dav':
            raddiiorhalfdav = np.array(davraddii[index])/2.
        if davradius == 'radius':
            raddiiorhalfdav = np.array(davraddii[index])
        result = avradius(positions_i, raddiiorhalfdav, useradius=useradius)
        new_data.append(result)
        if index % 50 == 0:
            print("    concluded %3.1f%%" % (100*index/len(positions)))
    print("    concluded %3.1f%%" % (100))

    return np.array(new_data)


def mine_bags(classes, bags, classesn='', bagsn='', operators=[np.average],  # Versao Nova
              operatorsn=['av']):
    """It mine regular features from atomic properties (bags) and atomic
    classes (also bags).

    The new regular features name are based in the
    new regular features name: "reg_" + oname + "_" + bname + "_" + cname,
    where oname, bname, and cname are elements of operatorsn, bagsn, and
    classesn, respectively.

    Parameter
    ---------
    bags: list with the bags to be examinated
    bagsn: list with the names of bags to be examinated
    classes: list of tuples with name and list of classes to be examinated
    classesn: class_name in the final feature
    operators: operation over the bag[class] array, (only applied if the class
               is not empth)
    operatorsn: operator name in the final feature
    sumclass: soma a quantidade de elementos para cada classe.

    Return
    ------
    list_of_new_features_name, list_of_new_features_data
    new_feature_name = "reg_" + operatorsn[i] + "_" + bagsn[j] + "_" + classesn[k]
    """

    print("Initializing minebags.")

    # ao longo da funcao serao adicionado os novos dados e nomes deles nessas
    # duas listas:
    list_of_new_features_name = []
    list_of_new_features_data = []

    # criando os novos features
    for operation, oname in zip(operators, operatorsn):
        for bag, bname in zip(bags, bagsn):
            for classa, cname in zip(classes, classesn):
                # criando nome do novo feature
                new_feature_name = "reg_" + oname + "_" + bname + "_" + cname
                # criando uma lista com o nome do feature para guardar os dados
                new_feature_data = []
                for sampleind in range(len(bag)):
                    classdata = np.array(classa[sampleind], dtype=bool)
                    if sum(classdata) == 0:
                        # if no one atom belongs to the classa
                        new_feature_data.append(np.nan)
                    else:
                        # if there are one atom which belongs to the classa
                        data = np.array(bag[sampleind], dtype=float)
                        reg_data_to_sampel = operation(data[classdata])
                        new_feature_data.append(reg_data_to_sampel)
                list_of_new_features_name.append(new_feature_name)
                list_of_new_features_data.append(new_feature_data.copy())

    return list_of_new_features_name, list_of_new_features_data


def classes_count(classes, classesn):  # Versao Nova   # preciso verificar se as classes com 0 elementos tao contando ou se ta resultando 0
    """It mix classes (bags) with an logical "and" to extract more classes, for
    each possible pair of classes in two lists.

    Parameters
    ----------
    classes: lists of pandas.Series.
             List with the class which will be counted.

    classesn: lists of str.
              Each str in this list is part of the name of the new reg data
              final name: 'reg_N_' + classesn

    Return
    ------
    comblist_of_new_features_name: list of str
                                   The new classes names:
                                   'bag_N_' + classesn

    list_of_new_features_data: list with new data
                               The new classes.
    """

    print('Initializing classes_count.')

    list_of_new_features_name = []
    list_of_new_features_data = []

    for clas, clasn in zip(classes, classesn):
        new_feature_name = "reg_N_" + clasn
        new_feature_data = []
        for index in range(len(clas)):
            classdata = np.sum(clas[index])
            new_feature_data.append(classdata)
        list_of_new_features_name.append(new_feature_name)
        list_of_new_features_data.append(new_feature_data.copy())

    return list_of_new_features_name, list_of_new_features_data


