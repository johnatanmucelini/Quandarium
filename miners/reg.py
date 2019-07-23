"""This module present functions that extract data from Quantum Chemistry (QC)
Calculations, and related functions. The QC codes with extractors avaliable
are:
- FHI-aims
- VESTA (not yet)
"""

import logging
import pandas as pd
import numpy as np
from quandarium.analy.aux import bag2arr
from quandarium.analy.aux import arr2bag

logging.basicConfig(filename='/home/johnatan/quandarium_module.log',
                    level=logging.INFO)
logging.info('The logging level is INFO')


# relative energies
def relative_energy(pd_df, relativefeature='reg_energy_tot',
                    relativefeaturen='reg_erel',
                    groupbyfeature='reg_chemical_formula'):
    """It calculate the relative energy for each chemical formula. Indeed, it
    calculate the relative feature for the groups of samples with the same
    valeu for the groupby feature. In other words, the complete dataset will be
    divided in n smaller dataset where the samples divide the same value of
    groupbyfeature, then the relative feature values will be calculated as the
    value of the featuretorelativize minus the smallest value of this feature
    in the group that it belongs to. Thus, any relative feuatere can be
    calculated. The new regular feature return in a new dataframe, concatenate
    with the input dataframe.

    Parameters
    ----------
    pd_df: pandas.DataFrame.
           A padas dataframe with coluns with the names in parameter
           relativefeature and groupbyfeature.
    relativefeature: str, (otional, default='reg_energy_tot')
                     The name of the column in the dataframe of the main
                     feature which will be compute the relative features.
    groupbyfeature: str, (otional, default='reg_chemical_formula')
                    The name of the column in the dataframe which will be
                    employed to create the groups of samples.
    Return
    ------
    combined_df: pandas.Datafram.
                 The input dataframe concatenated with data obtained.
    """
    chemical_compositions = np.unique(pd_df[[groupbyfeature]].values)
    relative_energies = np.zeros(len(pd_df))
    for chemical_compostion in chemical_compositions:
        isserie = pd_df[groupbyfeature] == chemical_compostion
        pd_s_etot = pd_df.loc[isserie, relativefeature].astype(float).values
        pd_s_etotmin = pd_s_etot.min()
        relative_energies[isserie] = pd_s_etot - pd_s_etotmin
    new_df = pd.DataFrame(np.array([relative_energies]).T,
                          columns=[relativefeaturen])
    combined_df = pd.concat([pd_df, new_df], axis=1, sort=False)
    return combined_df


def mine_bags(pd_df, classes, bags, classesn='', bagsn='', operators=[np.average],
              operatorsn=['av'], sumclass=''):
    """It mine regular features from atomic properties (bags) and atomic
    classes (also bags).

    The new regular features name are based in the
    new regular features name: "reg_" + oname + "_" + bname + "_" + cname,
    where oname, bname, and cname are elements of operatorsn, bagsn, and
    classesn, respectively.

    Parameter
    ---------
    pd_df: pandas data fragment
    bags: list with the bags to be examinated
    bagsn: list with the names of bags to be examinated
    classes: list of tuples with name and list of classes to be examinated
    classesn: class_name in the final feature
    operators: operation over the bag[class] array,
    operatorsn: operator name in the final feature
    sumclass: soma a quantidade de elementos para cada classe."""

    if classesn == '':
        classesn = []
        for classa in classes:
            classesn.append(classa.replace('bag_', ''))

    if bagsn == '':
        bagsn = []
        for bag in bags:
            bagsn.append(bag.replace('bag_', ''))

    if sumclass == '':
        logging.info('The sumclass was not seted, so, it will be automaticaly '
                     'set now.')
        sumclass = True
        for column in pd_df.columns.tolist():
            if 'reg_qtn_ce' in column:
                sumclass = False
                logging.info('The sumclass was automaticaly seted False: '
                             'regular data of the quantity of atoms in each'
                             'class (reg_qtn_ceXX) will not be calculated.')
                break


    # ao longo da funcao serao adicionado os novos dados e nomes deles nessas
    # duas listas:
    list_of_new_features_name = []
    list_of_new_features_data = []

    # Verificando quais bags não estão cheias
    for sampleind in range(len(pd_df)):
        for bag in bags:
            data = bag2arr(pd_df[bag][sampleind], dtype=float)
            if len(data) != 45:  # problem...
                print(sampleind, bag, data)
        for classa in classes:
            data = bag2arr(pd_df[classa][sampleind], dtype=bool)
            if len(data) != 45:  # problem...
                print(sampleind, classa, data)

    # criando os novos features
    for operation, oname in zip(operators, operatorsn):
        for bag, bname in zip(bags, bagsn):
            for classa, cname in zip(classes, classesn):
                # criando nome do novo feature
                new_feature_name = "reg_" + oname + "_" + bname + "_" + cname
                # criando uma lista com o nome do feature para guardar os dados
                new_feature_data = []
                for sampleind in range(len(pd_df)):
                    classdata = bag2arr(pd_df[classa][sampleind], dtype=bool)
                    if sum(classdata) == 0:
                        # if no one atom belongs to the classa
                        new_feature_data.append(np.nan)
                    else:
                        # if there are one atom which belongs to the classa
                        data = bag2arr(pd_df[bag][sampleind], dtype=float)
                        reg_data_to_sampel = operation(data[classdata])
                        new_feature_data.append(reg_data_to_sampel)

                list_of_new_features_name.append(new_feature_name)
                list_of_new_features_data.append(new_feature_data.copy())

    if sumclass:
        print(bags)
        # contando as quantidades de atomos em cada classe
        for classa, cname in zip(classes, classesn):
            new_feature_name = "reg_qtn_" + cname
            new_feature_data = []
            for sampleind in range(len(pd_df)):
                classdata = bag2arr(pd_df[classa][sampleind], dtype=bool)
                new_feature_data.append(sum(classdata))
            list_of_new_features_name.append(new_feature_name)
            list_of_new_features_data.append(new_feature_data.copy())

    # criando um pd.DataFrame que tem todos os dados e nome dos mesmos
    new_df = pd.DataFrame(np.array(list_of_new_features_data).T,
                          columns=list_of_new_features_name)

    combined_df = pd.concat([pd_df, new_df], axis=1, sort=False)
    return combined_df
