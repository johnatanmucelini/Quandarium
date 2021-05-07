import sys
import time
import multiprocessing as mp
import numpy as np

#sys.path.append('~/Quandarium/quandarium/')
from quandarium.aux import to_nparray
from quandarium.aux import to_list
from quandarium.mols import ecndav
from quandarium.mols import ecndav_rsopt
from quandarium.mols import ecndav_ropt
from quandarium.mols import findsc


def rec_ecndav_rsopt(kinfo, Rinfo, positions, cheme, print_convergence=False,
                     roundpijtoecn=False, w=''):
    """Return the effective coordination number (ecn), the average bound
    distance (dav), the optimized radius (ropt), and the conective index matrix
    Pij for each structure in the input pandas dataframe. See function
    quandarium.analy.mols.ecndav_rsopt.

    Parameters
    ----------
    kinfo: float, np.array, or dictionary.
           Coordination activation factor, a float will give the same factor k
           for each possible coordination. A np.array (n,n) shaped, where n is
           the quantity of atoms, will be consider the each entry as the k for
           each pair of possible coordination. A dictionary will construct
           each k factor as dict[cheme[i]] plus dict[cheme[j]].
    Rinfo: np.array or a dict.
           The atomic tabeled radius. If a dict, each element radius will be
           consider as dict[cheme[i]].
    positionsfeature: str (optional, default='bag_positions')
                      The name of the fuature (bag type) in pd_df with
                      cartezian positions of the atoms.
    chemefeature: str (optional, default='bag_positions')
                  The name of the fuature (bag type) in pd_df with chemical
                  elements of each atoms.
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

    print("Initializing analysis: rec_ecndav_rsopt")

    positions = to_nparray(positions)
    cheme = to_nparray(cheme)
    list_bag_ecn = []
    list_bag_dav = []
    list_bag_ori = []
    list_bag_pij = []
    for index in range(len(positions)):
        cheme_i = cheme[index]
        positions_i = positions[index]
        if w:
            ecn, dav, ori, pij = ecndav_rsopt(positions_i, cheme_i, kinfo, Rinfo,
                                              roundpijtoecn=roundpijtoecn, w=w)
        else:
            ecn, dav, ori, pij = ecndav_rsopt(positions_i, cheme_i, kinfo, Rinfo,
                                              roundpijtoecn=roundpijtoecn)
        #                                 print_convergence=print_convergence,
        list_bag_ecn.append(to_list(ecn))
        list_bag_dav.append(to_list(dav))
        list_bag_ori.append(to_list(ori))
        list_bag_pij.append(to_list(pij))
        if index % 50 == 0 and print_convergence:
            print("    concluded %3.1f%%" % (100*index/len(positions)))
    if print_convergence:
        print("    concluded %3.1f%%" % (100))
    list_of_new_features_data = [list_bag_ecn, list_bag_dav, list_bag_ori,
                                 list_bag_pij]
    list_of_new_features_name = ['bag_ecn_rsopt', 'bag_dav_rsopt',
                                 'bag_ori_rsopt', 'bag_pij_rsopt']

    return list_of_new_features_name, list_of_new_features_data


def rec_ecndav_ropt(positions, cheme, print_convergence=False,  # Versao Nova
                    roundpijtoecn=False):
    """Return the effective coordination number (ecn), the average bound
    distance (dav), the optimized radius (ropt), and the conective index matrix
    Pij for each structure in the inputs.
    See also:
    quandarium.analy.mols.ecndav_ropt

    Parameters
    ----------
    positions: np.array, pandas.Series, or list
               Several datas with the atoms positions in bags of bags =
               [[x1,y1,z1], [x2,y2,z2], ...] .
    cheme: np.array, pandas.Series, or list
           Several datas of the chemical elements of each atoms.
    print_convergence: boolean, (optional, default=False).
                       It treu, the convergency will be printed.
    Returns
    -------
    bag_ecn, bag_dav: bag of floats, with arrays.
                      They contain the calculated ecn and dav for each atom.
    bag_of_bag_pij: bag of bag of floats, with (n,n) shaped arrays.
                    The index of connectivity between pairs of atoms.
    """
    positions = to_nparray(positions).tolist()
    cheme = to_nparray(cheme).tolist()
    print("Initializing analysis: rec_ecndav_ropt")
    list_bag_ecn = []
    list_bag_dav = []
    list_bag_ori = []
    list_bag_pij = []
    for index in range(len(positions)):
        cheme_i = np.array(cheme[index])
        positions_i = np.array(positions[index])
        ecn, dav, ori, pij = ecndav_ropt(positions_i, cheme_i, plot_name='',
                                         print_convergence=print_convergence,
                                         roundpijtoecn=roundpijtoecn)
        list_bag_ecn.append(ecn)
        list_bag_dav.append(dav)
        list_bag_ori.append(ori)
        list_bag_pij.append(pij)
        if index % 50 == 0:
            print("    concluded %3.1f%%" % (100*index/len(positions)))
    print("    concluded %3.1f%%" % (100))
    list_of_new_features_data = [list_bag_ecn, list_bag_dav, list_bag_ori,
                                 list_bag_pij]
    list_of_new_features_name = ['bag_ecn_ropt', 'bag_dav_ropt',
                                 'bag_ori_ropt', 'bag_pij_ropt']
    return list_of_new_features_name, list_of_new_features_data


def rec_ecndav(positions, print_convergence=False):   # Versao nova / Não paralelizado
    """Return the effective coordination number (ecn) and the average bound
    distance and the conective index matrix Pij for each structure in the input
    pandas dataframe. See function quandarium.analy.mols.ecndav.

    Parameters
    ----------
    positions: np.array, pandas.Series, or list
               Several datas with the atoms positions in bags of bags =
               [[x1,y1,z1], [x2,y2,z2], ...] .
    print_convergence: boolean, (optional, default=False).
                       It treu, the convergency will be printed.
    Returns
    -------
    list_of_new_features_name: list.
                               ['bag_ecn', 'bag_dav', 'bag_pij']
    list_of_new_features_data: list.
                               lists with the ecn, dav, pij data for each atom.
    """

    print("Initializing analysis: rec_ecndav")

    positions = to_nparray(positions).tolist()
    list_bag_ecn = []
    list_bag_dav = []
    list_bag_pij = []
    for index in range(len(positions)):
        positions_i = np.array(positions[index])
        ecn, dav, pij = ecndav(positions_i, print_convergence=print_convergence)
        list_bag_ecn.append(to_list(ecn))
        list_bag_dav.append(to_list(dav))
        list_bag_pij.append(to_list(pij))
        if index % 50 == 0:
            print("    concluded %3.1f%%" % (100*index/len(positions)))
    print("    concluded %3.1f%%" % (100))
    list_of_new_features_data = [list_bag_ecn, list_bag_dav, list_bag_pij]
    list_of_new_features_name = ['bag_ecn', 'bag_dav', 'bag_pij']

    return list_of_new_features_name, list_of_new_features_data


def findsc_wrap(inputs):  #It should be here
    """Wraper to multiprocessing findsc"""
    result = findsc(positions=inputs[0], atomic_radii=inputs[1],
                    adatom_radius=inputs[2], remove_is=inputs[3],
                    ssamples=inputs[4], writw_sp=inputs[5],
                    return_expositions=inputs[6],
                    print_surf_properties=inputs[7], sp_file=inputs[8])
    return result


# New version, paralelised

def rec_findsc(positions, davraddi, davradius='dav', adatom_radius=1.1,  
               ssamples=1000, return_expositions=True,
               print_surf_properties=False, remove_is=True, procs=1):
    """It return the atom site surface(True)/core(Flase) for each atoms in
    for each structure pandas dataframe. See more in the function
    quandarium.analy.mols.findsurfatons.

    Parameters
    ----------
    positions: Pandas.Series 
               The name of the fuature (bag type) in pd_df with
               cartezian positions of the atoms.
    davraddi:  Pandas.Series
               The name of the fuature in pd_df with atomic radii or dav
               information (bag of floats).
    davradius: str ['dav','radii'] (optional, default='dav')
               If radii, atomic radius will be the feature davraddiifeature
               values. If dav the values in atomic radius will be half of the
               feature davraddiifeature values.
    adatom_radius: float (optional, default=1.1).
                   Radius of the dummy adatom, in angstroms.               
    ssampling: intiger (optional, default=1000).
               Quantity of samplings over the touched sphere surface of each
               atom.

    Return
    ------
    list_of_new_features_name: list with strings.
                               ['bag_issurface', 'bag_surfaceexposition']
    list_of_new_features_data: list with data.
                               issurface: bag of intiger
                                          The number indicate 1 to surface
                                          atoms, 0 to core atoms.
                               surfaceexposition: bag of floats.
                                                  The percentual of surface
                                                  exposition of each atom.
    """

    print("Initializing analysis: rec_findsc")

    inputs_list = []
    list_is_surface = []
    list_exposition = []
    for index, (poitionsi, davraddii) in enumerate(zip(positions, davraddi)):
        #print(type(poitionsi),poitionsi)
        positionsi = np.array(poitionsi)  # manter np.array e ativar bags
        if davradius == 'dav':
            atomic_radii = np.array(davraddii)/2  # manter np.array e ativar bags
        if davradius == 'radii':
            atomic_radii = np.array(davraddii)
        inputs_list.append([positionsi, atomic_radii, adatom_radius,
                            remove_is, ssamples, False, return_expositions,
                            print_surf_properties, "surface_points.xyz"])
    pool = mp.Pool(procs)
    result = pool.map_async(findsc_wrap, inputs_list, chunksize=1)

    while not result.ready():
        remaining = result._number_left  # pylint: disable=W0212
        print('Remaining: ', remaining)
        time.sleep(5.0)
    print('Finished')

    outputs = result.get()
    for index, _ in enumerate(outputs):
        list_is_surface.append(outputs[index][0])
        list_exposition.append(outputs[index][1])
    list_of_new_features_data = [list_is_surface, list_exposition]
    list_of_new_features_name = ['bag_issurf', 'bag_exposition']

    return list_of_new_features_name, list_of_new_features_data


def class_from_bins(list_of_bags, list_of_binsvals, list_of_new_classes_names):  # Versao nova
    """It take a class for bins of atomic properties for a bags.
    list_of_bags: list of bags datas

    list_of_binsvals: list with list of bins

    list_of_new_classes_names: list with list of new classes
    """

    print("Initializing class_from_bins.")

    list_of_new_classes_data = []
    list_of_new_classes_name = []
    for bag, binsvals, new_class_names in zip(list_of_bags, list_of_binsvals,
                                              list_of_new_classes_names):
        for bindex, name in enumerate(new_class_names):
            list_of_new_classes_name.append(name)
            new_class_data = []
            for index in range(len(bag)):
                bagdata = np.array(bag[index])
                c1 = bagdata >= binsvals[bindex]
                c2 = bagdata < binsvals[bindex + 1]
                result = np.logical_and(c1, c2)
                new_class_data.append(result.tolist())
            list_of_new_classes_data.append(new_class_data)

    return list_of_new_classes_name, list_of_new_classes_data


def classes_from_dvalues(bags, classesbasen, classesvals, classesvalsn):  # Nova
    """It take classes from discrete values of other features (bags):

    new_class_name(i,j): "bag_" + classesbasen[i] + classesvalsn[i][j]
    new_class_data(i,j): bags[i] == classesvals[i][j]

    wherer, i run over the bags indexes, j rum over the classes discrete values
    indexes, and s run over the samples indexes.

    Parameters
    ----------
    bags: list of pandas.Series.
          Bags features names which will be analysed to extrac classes from its
          values.
    classesbasen: list of str.
                  If it is a list with str, aech element is the base name of
                  the new class, with will be extract from the respective bag.
    classesvals: list of list of values.
                 Each list inside it present a list the discrete values of the
                 bags cosidered for each new class.
    classesvalsn: list of lists with str.
                  Each list inside it present part of the names of the new
                  class which will be obtained from the respective bag in bags.

    Return
    ------
    list_of_new_classes_name: list of str.
                              The combination of the names inputs:
                              "bag_" + classesbasen[i] + classesvalsn[i][j]
    list_of_new_classes_data: list of data
                              The new data obtained
    """

    print("Initializing classes_from_dvalues.")

    # Ao longo da funcao serao adicionado os novos dados e nomes deles nessas
    # duas listas:
    list_of_new_classes_name = []
    list_of_new_classes_data = []

    for bag, classbasen, vals, valsn in zip(bags, classesbasen, classesvals,
                                            classesvalsn):
        for val, valn in zip(vals, valsn):
            # criando nome do novo feature
            new_class_name = "bag_" + classbasen + valn
            # criando uma lista com o nome do feature para guardar os dados
            new_class_data = []
            for index in range(len(bag)):
                bagdata = np.array(bag[index])
                classdata = bagdata == val
                # print(classdata,bagdata,val)
                new_class_data.append(classdata.tolist())
            list_of_new_classes_name.append(new_class_name)
            list_of_new_classes_data.append(new_class_data.copy())

    return list_of_new_classes_name, list_of_new_classes_data


def classes_mixing(classes1, classes2, classesn1='', classesn2=''):  # Versao Nova
    """It mix classes (bags) with an logical "and" to extract more classes, for
    each possible pair of classes in two lists.

    Parameters
    ----------
    classes1, classes2: lists of pandas.Series.
                        List with the features (bag, class) which will be
                        mixtured to obtain more classes.
                        new class: classes1[i] == classes2[j]

    classesn1, classesn2: lists of str.
                          Each str in this list is part of the name in the
                          final name: 'bag_' + classesn1[i] + classesn2[j]

    Return
    ------
    comblist_of_new_features_name: list of str
                                   The new classes names:
                                   'bag_' + classesn1[i] + classesn2[j]

    list_of_new_features_data: list with new data
                               The new classes.
    """

    print('Initializing classes_mixing.')

    list_of_new_features_name = []
    list_of_new_features_data = []

    for class1, classn1 in zip(classes1, classesn1):
        for class2, classn2 in zip(classes2, classesn2):
            # criando nome do novo feature
            new_feature_name = "bag_" + classn1 + classn2
            # criando uma lista com o nome do feature para guardar os dados
            new_feature_data = []
            for index in range(len(classes1[0])):
                class1data = np.array(class1[index], dtype=bool)
                class2data = np.array(class2[index], dtype=bool)
                finalclassdata = np.logical_and(class1data, class2data)
                new_feature_data.append(finalclassdata.tolist())
            list_of_new_features_name.append(new_feature_name)
            list_of_new_features_data.append(new_feature_data.copy())

    return list_of_new_features_name, list_of_new_features_data
