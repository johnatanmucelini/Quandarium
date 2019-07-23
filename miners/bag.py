"""This module present functions that extract data from Quantum Chemistry (QC)
Calculations, and related functions. The QC codes with extractors avaliable
are:
- FHI-aims
- VESTA (not yet)
"""

import logging
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
from quandarium.analy.aux import bag2arr
from quandarium.analy.aux import arr2bag
from quandarium.analy.mols import ecndav
from quandarium.analy.mols import ecndav_ropt
from quandarium.analy.mols import findsc

logging.basicConfig(filename='/home/johnatan/quandarium_module.log',
                    level=logging.INFO)
logging.info('The logging level is INFO')


def rec_ecndav_ropt(pd_df, positionsfeature='bag_positions',
                    chemefeature='bag_cheme', print_convergence=False):
    """Return the effective coordination number (ecn), the average bound
    distance (dav), the optimized radius (ropt), and the conective index matrix
    Pij for each structure in the input pandas dataframe. See function
    quandarium.analy.mols.ecndav_ropt.

    Parameters
    ----------
    pd_df: pandas.DataFrame.
           A pandas dataframe with all the positions needed to the analysis
    positionsfeature: str (optional, default='bag_positions')
                      The name of the fuature (bag type) in pd_df with
                      cartezian positions of the atoms.
    print_convergence: boolean, (optional, default=False).
                       It treu, the convergency will be printed.
    Returns
    -------
    combined_df: pandas.DataFrame.
                 The combination of the input dataframe plus a dataframe with
                 more three features:
                 bag_ecn, bag_dav: bag of floats, (n,) shaped.
                                   They contain the calculated ecn and dav for
                                   each atom.
                 bag_of_bag_pij: bag of bag of floats, (n,n) shaped.
                                 The index of connectivity between pairs of
                                 atoms.
    """

    print("Initializing analysis: rec_ecndav_ropt")
    list_bag_ecn = []
    list_bag_dav = []
    list_bag_ori = []
    list_bag_pij = []
    for index in range(len(pd_df)):
        cheme = bag2arr(pd_df[chemefeature][index])
        positions = bag2arr(pd_df[positionsfeature][index])
        ecn, dav, ori, pij = ecndav_ropt(positions, cheme, plot_name='',
                                         print_convergence=False)
        list_bag_ecn.append(arr2bag(ecn))
        list_bag_dav.append(arr2bag(dav))
        list_bag_ori.append(arr2bag(ori))
        list_bag_pij.append(arr2bag(pij))
        if index % 50 == 0:
            print("    concluded %3.1f%%" % (100*index/len(pd_df)))
    print("    concluded %3.1f%%" % (100))
    list_of_new_features_data = [list_bag_ecn, list_bag_dav, list_bag_ori,
                                 list_bag_pij]
    list_of_new_features_name = ['bag_ecn_ropt', 'bag_dav_ropt',
                                 'bag_ori_ropt', 'bag_pij_ropt']

    # Creating and combinating the pandas DataFrame
    new_df = pd.DataFrame(np.array(list_of_new_features_data).T,
                          columns=list_of_new_features_name)
    combined_df = pd.concat([pd_df, new_df], axis=1, sort=False)

    return combined_df


def rec_ecndav(pd_df, positionsfeature='bag_positions',
                print_convergence=False):
    """Return the effective coordination number (ecn) and the average bound
    distance and the conective index matrix Pij for each structure in the input
    pandas dataframe. See function quandarium.analy.mols.ecndav.

    Parameters
    ----------
    pd_df: pandas.DataFrame.
           A pandas dataframe with all the positions needed to the analysis
    positionsfeature: str (optional, default='bag_positions')
                      The name of the fuature (bag type) in pd_df with
                      cartezian positions of the atoms.
    print_convergence: boolean, (optional, default=False).
                       It treu, the convergency will be printed.
    Returns
    -------
    combined_df: pandas.DataFrame.
                 The combination of the input dataframe plus a dataframe with
                 more three features:
                 bag_ecn, bag_dav: bag of floats, (n,) shaped.
                                   They contain the calculated ecn and dav for
                                   each atom.
                 bag_of_bag_pij: bag of bag of floats, (n,n) shaped.
                                 The index of connectivity between pairs of
                                 atoms.
    """

    print("Initializing analysis: rec_ecndav")
    list_bag_ecn = []
    list_bag_dav = []
    list_bag_pij = []
    for index in range(len(pd_df)):

        positions = eval('np.array(' + pd_df[positionsfeature][index] + ')')
        ecn, dav, pij = ecndav(positions, print_convergence=print_convergence)
        list_bag_ecn.append(arr2bag(ecn))
        list_bag_dav.append(arr2bag(dav))
        list_bag_pij.append(arr2bag(pij))
        if index % 50 == 0:
            print("    concluded %3.1f%%" % (100*index/len(pd_df)))
    print("    concluded %3.1f%%" % (100))
    list_of_new_features_data = [list_bag_ecn, list_bag_dav, list_bag_pij]
    list_of_new_features_name = ['bag_ecn', 'bag_dav', 'bag_pij']

    # Creating and combinating the pandas DataFrame
    new_df = pd.DataFrame(np.array(list_of_new_features_data).T,
                          columns=list_of_new_features_name)
    combined_df = pd.concat([pd_df, new_df], axis=1, sort=False)

    return combined_df


def symatoms(pd_df, bags, chemefeature=''):
    """Takes an index of symetry based in bags information"""

    qtnsamples = len(pd_df)
    matrixsym = np.zeros([qtnsamples, qtnsamples])

    # getting minmax to normalize the bags
    bagmin = []
    bagmaxmindiff = []
    for bag in bags:
        data = []
        for sampind in range(qtnsamples):
            data.append(bag2arr(pd_df[bag].values[sampind], dtype=float))
        data = np.array([data])
        bagmin.append(data.min())
        if data.max() - data.min() == 0.:
            bagmaxmindiff.append(1.)
        else:
            bagmaxmindiff.append(data.min() - data.max())


    for samp1ind in range(qtnsamples):
        for samp2ind in range(qtnsamples):
            if samp1ind > samp2ind:
                # getting information from bags
                costmatrix = np.zeros([45,45])
                for bagind, bag in enumerate(bags):
                    bagsamp1data = bag2arr(pd_df[bag].values[samp1ind], dtype=float)
                    bagsamp2data = bag2arr(pd_df[bag].values[samp2ind], dtype=float)
                    bagsamp1data = (bagsamp1data - bagmin[bagind]) / bagmaxmindiff[bagind]
                    bagsamp2data = (bagsamp2data - bagmin[bagind]) / bagmaxmindiff[bagind]
                    samp1info = np.array([bagsamp1data])
                    samp2info = np.array([bagsamp2data])
                    costmatrix += (samp1info - samp2info.T)**2
                rowind, colind = linear_sum_assignment(costmatrix)
                cost = costmatrix[rowind, colind].sum()
                matrixsym[samp1ind, samp2ind] = cost
                matrixsym[samp2ind, samp1ind] = cost
                print('concluded')
    return matrixsym

def rec_findsc(pd_df, adatom_radius, positionsfeature='bag_positions',
               davraddiifeature='bag_dav', davradius='dav',
               ssamples=1000, return_expositions=True,
               print_surf_properties=False, remove_is=True):
    """It return the atom site surface(True)/core(Flase) for each atoms in
    for each structure pandas dataframe. See more in the function
    quandarium.analy.mols.findsurfatons.

    Parameters
    ----------
    pd_df: pandas.DataFrame.
           A pandas dataframe with all the positions needed to the analysis
    adatom_radius: float (optional, default=1.1).
                   Radius of the dummy adatom, in angstroms.
    positionsfeature: str (optional, default='bag_positions')
                      The name of the fuature (bag type) in pd_df with
                      cartezian positions of the atoms.
    davraddiifeature: str (optional, default='bag_dav')
                      The name of the fuature in pd_df with atomic radii or dav
                      information (bag of floats).
    davradius: str ['dav','radii'] (optional, default='dav')
               If radii, atomic radius will be the feature davraddiifeature
               values. If dav the values in atomic radius will be half of the
               feature davraddiifeature values.
    ssampling: intiger (optional, default=1000).
               Quantity of samplings over the touched sphere surface of each
               atom.

    Return
    ------
    combined_df: pandas.DataFrame.
                 The combination of the input dataframe plus a dataframe with
                 more two features:
                 is_surface: bag of intiger
                             The number indicate 1 to surface atoms, 0 to core
                             atoms.
                 surface_exposition: bag of floats.
                                     The percentual of surface exposition of
                                     each atom.
    """

    print("Initializing analysis: rec_findsc")
    list_is_surface = []
    list_exposition = []
    list_bag_pij = []
    for index in range(len(pd_df)):
        positions = eval('np.array(' + pd_df[positionsfeature][index] + ')')
        if davradius == 'dav':
            atomic_radii = eval('np.array(' + pd_df[davraddiifeature][index]
                                + ')')/2
        if davradius == 'radii':
            atomic_radii = eval('np.array(' + pd_df[davraddiifeature][index]
                                + ')')
        is_surface, exposition = findsc(positions, atomic_radii,
                                        adatom_radius, remove_is=remove_is,
                                        ssamples=ssamples, writw_sp=True,
                                        return_expositions=return_expositions,
                                        print_surf_properties=print_surf_properties,
                                        sp_file="surface_points.xyz")
        list_is_surface.append(arr2bag(is_surface))
        list_exposition.append(arr2bag(exposition))
        if index % 50 == 0:
            print("    concluded %3.1f%%" % (100*index/len(pd_df)))
    print("    concluded %3.1f%%" % (100))
    list_of_new_features_data = [list_is_surface, list_exposition]
    list_of_new_features_name = ['bag_issurf', 'bag_exposition']

    # Creating and combinating the pandas DataFrame
    new_df = pd.DataFrame(np.array(list_of_new_features_data).T,
                          columns=list_of_new_features_name)
    combined_df = pd.concat([pd_df, new_df], axis=1, sort=False)

    return combined_df


def class_from_svalues(pd_df, bag, classesbasen='', classesvals='',
                         classesvalsn='', add_ones=True):
    """It take classes from specific values of a single feature bag:"""


def classes_from_dvalues(pd_df, bags, classesbasen='', classesvals='',
                         classesvalsn='', add_ones=True):
    """It take classes from discrete values of other features (bags):

    new_class_name(i,j): "bag_" + classesbasen[i] + classesvalsn[i][j]
    new_class_data(i,j,s): bags[i][s] == classesvals[i][j]

    wherer, i run over the bags indexes, j rum over the classes discrete values
    indexes, and s run over the samples indexes.

    Parameters
    ----------
    pd_df: pandas.DataFrame.
           A pandas dataframe with the bags to extract classes.
    bags: lists of str.
          List with the bags features names which will be analysed to extrac
          classes from its values.
    classesbasen: list of str, or '' (optional, default='').
                  If it is a list with str, aech element is the base name of
                  the new class, with will be extract from the respective bag.
                  If '' the name will be automaticaly obtained from the names
                  of the bags (bag.replace('bag_', '').
    classesvals: list of lists with str, or '' (optional, default='').
                 If it is a list of lists with str, aech list inside it present
                 the discrete values of the bags cosidered for each new class.
                 If '' the name will be automaticaly obtained from each unique
                 values of the bags (considering the completely dataset).
    classesvalsn: list of lists with str, or '' (optional, default='').
                  If it is a list of lists with str, aech list inside it
                  present the names of the new class which will be obtained
                  from the respective bag in bags. If '' the name will be
                  automaticaly obtained from each classesvals (independent if
                  it was automaticaly builted), in this case, "." are removed
                  from the classes names.
    Return
    ------
    combined_df: pandas.DataFrame.
                 The combination of the input dataframe plus a dataframe with
                 the new classes.
    """

    if classesbasen == '':
        classesbasen = []
        for bag in bags:
            classesbasen.append(bag.replace('bag_', ''))

    if classesvals == '':
        classesvals = []
        for bag in bags:
            jointedbag = ','.join(pd_df[bag].values)
            jointedbagaux = jointedbag.replace('[', '').replace(']', '')
            unique_vals = np.unique(jointedbagaux.split(',')).tolist()
            classesvals.append(unique_vals)

    if classesvalsn == '':
        classesvalsn = []
        for unique_vals in classesvals:
            classesvalsn.append([])
            for val in unique_vals:
                valn = str(val).replace('.', '')
                classesvalsn[-1].append(valn)

    # ao longo da funcao serao adicionado os novos dados e nomes deles nessas
    # duas listas:
    list_of_new_classes_name = []
    list_of_new_classes_data = []

    # Verificando quais bags não estão cheias
    for sampleind in range(len(pd_df)):
        for bag in bags:
            data = bag2arr(pd_df[bag][sampleind], dtype=str)
            if len(data) != 45:  # problem...
                print(sampleind, bag, data)

    for bag, classbasen, vals, valsn in zip(bags, classesbasen, classesvals,
                                            classesvalsn):
        for val, valn in zip(vals, valsn):
            # criando nome do novo feature
            new_class_name = "bag_" + classbasen + valn
            # criando uma lista com o nome do feature para guardar os dados
            new_class_data = []
            for sampleind in range(len(pd_df)):
                bagdata = bag2arr(pd_df[bag][sampleind])
                classdata = bagdata == val
                #print(classdata,bagdata,val)
                new_class_data.append(arr2bag(classdata))
            list_of_new_classes_name.append(new_class_name)
            list_of_new_classes_data.append(new_class_data.copy())

    if add_ones:
        data = []
        for ind in range(len(pd_df)):
            onessize = len(bag2arr(pd_df[bags[0]][ind]))
            ones = np.ones(onessize, dtype=bool)
            data.append(arr2bag(ones))
        list_of_new_classes_name.append('bag_ceALL')
        list_of_new_classes_data.append(data.copy())

    # criando um pd.DataFrame que tem todos os dados e nome dos mesmos
    new_df = pd.DataFrame(np.array(list_of_new_classes_data).T,
                          columns=list_of_new_classes_name)

    combined_df = pd.concat([pd_df, new_df], axis=1, sort=False)
    return combined_df


def classes_mixing(pd_df, classes1, classes2, classesn1='', classesn2=''):
    """It mix classes (bags) with an logical "and" to extract more classes, for
    each possible pair of classes in two lists.

    Parameters
    ----------
    pd_df: pandas.DataFrame.
           A pandas dataframe with all the classes needed to the analysis.
    classes1, classes2: lists of str.
                        List with the features (bags!) which will be mixtured
                        to obtain more classes.

    classesn1, classesn2: lists of str or '' (optional, default='').
                          If it is a list, aech value is the name in the final
                          variable of each class in classes1 and classes2. If
                          '' the name will be automaticaly obtained from each
                          classes1 names, by removing 'bag_' from its names.

    Return
    ------
    combined_df: pandas.DataFrame.
                 The combination of the input dataframe plus a dataframe with
                 the new classes.
    """

    if classesn1 == '':
        classesn1 = []
        for class1 in classes1:
            classesn1.append(class1.replace('bag_', ''))
    if classesn2 == '':
        classesn2 = []
        for class2 in classes2:
            classesn2.append(class2.replace('bag_', ''))

    # ao longo da funcao serao adicionado os novos dados e nomes deles nessas
    # duas listas:
    list_of_new_features_name = []
    list_of_new_features_data = []

    # Verificando quais bags não estão cheias
    for sampleind in range(len(pd_df)):
        for classa in classes1 + classes2:
            data = bag2arr(pd_df[classa][sampleind], dtype=float)
            if len(data) != 45:  # problem...
                print(sampleind, classa, data)

    for class1, classn1 in zip(classes1, classesn1):
        for class2, classn2 in zip(classes2, classesn2):
            # criando nome do novo feature
            new_feature_name = "bag_" + classn1 + classn2
            # criando uma lista com o nome do feature para guardar os dados
            new_feature_data = []
            for sampleind in range(len(pd_df)):
                class1data = bag2arr(pd_df[class1][sampleind], dtype=bool)
                class2data = bag2arr(pd_df[class2][sampleind], dtype=bool)
                finalclassdata = np.logical_and(class1data, class2data)
                new_feature_data.append(arr2bag(finalclassdata))
            list_of_new_features_name.append(new_feature_name)
            list_of_new_features_data.append(new_feature_data.copy())

    # criando um pd.DataFrame que tem todos os dados e nome dos mesmos
    new_df = pd.DataFrame(np.array(list_of_new_features_data).T,
                          columns=list_of_new_features_name)

    combined_df = pd.concat([pd_df, new_df], axis=1, sort=False)
    return combined_df
