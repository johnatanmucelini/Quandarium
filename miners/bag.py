"""This module present functions that extract data from Quantum Chemistry (QC)
Calculations, and related functions. The QC codes with extractors avaliable
are:
- FHI-aims
- VESTA (not yet)
"""

import multiprocessing as mp
import time
import logging
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
from quandarium.analy.aux import to_nparray
from quandarium.analy.aux import bag2arr
from quandarium.analy.aux import to_list
from quandarium.analy.aux import arr2bag
from quandarium.analy.aux import logcolumns
from quandarium.analy.mols import ecndav
from quandarium.analy.mols import ecndav_rsopt
from quandarium.analy.mols import ecndav_ropt
from quandarium.analy.mols import findsc
from quandarium.analy.mols import connections


logging.basicConfig(filename='/home/johnatan/quandarium_module.log',
                    level=logging.INFO)
logging.info('The logging level is INFO')


def rec_connections(pd_df, baseucheme, positionsfeature='bag_positions',
                    chemefeature='bag_cheme', pijfeature='bag_pij', stype='bl',
                    dictcheme='', print_analysis=False):   # Versao Velha
    """This analysis seach for conectivities in the atoms neighborhood based if
    the chemical element cheme (or other discrete feature) for each structure
    in the input pandas dataframe. See function
    quandarium.analy.mols.connections.

    Parameters
    ----------
    pd_df: pandas.DataFrame.
           A pandas dataframe with all the positions needed to the analysis
    baseucheme: np.array of str.
                The base chemical elements to look for connections.
    positionsfeature: str (optional, default='bag_positions')
                      The name of the fuature (bag type) in pd_df with
                      cartezian positions of the atoms.
    chemefeature: str (optional, default='bag_positions')
                  The name of the fuature (bag type) in pd_df with chemical
                  elements of each atoms.
    pijfeature: str (optional, default='bag_pij')
                The name of the fuature (bag type) in pd_df with the weight of
                the conectivity between the atoms pairs.
    bag_cheme: str (optional, default='bag_cheme')
               The name of the fuature (bag type) in pd_df with chemical
               elements of the atoms.
    stype: str (optional, default='bl')
           A string determining the type of conection to seach for. 'bb'
           indicate back bonds, 'bc' indicate ciclic bonds, while bl indicate
           line bonds.
    print_analysis: boolean, (default=False)
                    It True, information for each analysis will be printed.
    Returns
    -------
    combined_df: pandas.DataFrame.
                 The combination of the input dataframe plus a dataframe with
                 a new features:
                 fd_connect: np.array (n,m) shaped were m is the number of
                             atoms types. The nearest (first degree) neighbors
                             connections types. All in alphabatic order. For
                             instance, in a molecule with atoms of types A an
                             B: [-A, -B].
                  sd_connect: np.array (n,m**2) shaped were m is the number of
                              atoms types. The second degree neighbors
                              connections. All in alphabatic order. For
                              instance, in a molecule with atoms of types A an
                              B: [-A-A, -A-B, -B-A, -B-B].
                  td_connect: np.array (n,m**3) shaped were m is the number of
                              atoms types. The third degree neighbors
                              connections. All in alphabatic order. For
                              instance, in a molecule with atoms of types A an
                              B: [-A-A-A, -A-A-B, -A-B-A, -A-B-B, -B-A-A,
                              -B-A-B, -B-B-A, -B-B-B].
    """

    print("Initializing analysis: rec_connections")
    logging.info('Initializing analysis: rec_connections')
    logging.info('positionsfeature: {}'.format(positionsfeature))
    logging.info('chemefeature: {}'.format(chemefeature))
    logging.info('pijfeature: {}'.format(pijfeature))
    logging.info('dictcheme: {}'.format(dictcheme))
    logging.info('baseucheme: {}'.format(baseucheme))
    logging.info('stype: {}'.format(stype))
    logging.info('print_analysis: {}'.format(print_analysis))
    # logging.info('print_convergence: {}'.format(print_convergence))

    list_bag_fdconnect = []
    list_bag_sdconnect = []
    list_bag_tdconnect = []
    for index in range(len(pd_df)):
        logging.info('    Proceding analysis of structure index {:04d}'.format(
            index))
        cheme = bag2arr(pd_df[chemefeature][index])
        positions = bag2arr(pd_df[positionsfeature][index])
        pij = bag2arr(pd_df[pijfeature][index])
        qtna = len(positions)
        qtnuc = len(baseucheme)
        fd_data = np.zeros([qtna, qtnuc])
        sd_data = np.zeros([qtna, qtnuc**2])
        td_data = np.zeros([qtna, qtnuc**3])
        if qtna > 1:
            fd_data, sd_data, td_data = connections(positions, cheme, pij,
                                                    stype=stype,
                                                    dictcheme=dictcheme,
                                                    baseucheme=baseucheme,
                                                    print_analysis=
                                                    print_analysis)
        list_bag_fdconnect.append(arr2bag(fd_data))
        list_bag_sdconnect.append(arr2bag(sd_data))
        list_bag_tdconnect.append(arr2bag(td_data))
        if index % 50 == 0:
            print("    concluded %3.1f%%" % (100*index/len(pd_df)))
    print("    concluded %3.1f%%" % (100))
    list_of_new_features_data = [list_bag_fdconnect, list_bag_sdconnect,
                                 list_bag_tdconnect]
    list_of_new_features_name = ['bag_fdconnect', 'bag_sdconnect',
                                 'bag_tdconnect']

    # Creating and combinating the pandas DataFrame
    new_df = pd.DataFrame(np.array(list_of_new_features_data).T,
                          columns=list_of_new_features_name)
    combined_df = pd.concat([pd_df, new_df], axis=1, sort=False)

    logcolumns('info', "New columns:", new_df)

    return combined_df


def rec_ecndav_rsopt(kinfo, Rinfo, positions, cheme, print_convergence=False,
                     roundpijtoecn=False, w=''):  # Versao Nova
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
    logging.info('Initializing analysis: rec_ecndav_rsopt')
    logging.info('kinfo: {}'.format(kinfo))
    logging.info('Rinfo: {}'.format(Rinfo))

    positions = to_nparray(positions)
    cheme = to_nparray(cheme)
    # logging.info('print_convergence: {}'.format(print_convergence))
    list_bag_ecn = []
    list_bag_dav = []
    list_bag_ori = []
    list_bag_pij = []
    for index in range(len(positions)):
        logging.info('    Proceding analysis of structure index {:04d}'.format(
            index))
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
        if index % 50 == 0:
            print("    concluded %3.1f%%" % (100*index/len(positions)))
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
        logging.info('    Proceding analysis of structure index {:04d}'.format(
            index))
        cheme_i = np.array(cheme[index])
        positions_i = np.array(positions[index])
        ecn, dav, ori, pij = ecndav_ropt(positions_i, cheme_i, plot_name='',
                                         print_convergence=print_convergence,
                                         roundpijtoecn=roundpijtoecn)
        list_bag_ecn.append(arr2bag(ecn))
        list_bag_dav.append(arr2bag(dav))
        list_bag_ori.append(arr2bag(ori))
        list_bag_pij.append(arr2bag(pij))
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
    logging.info('Initializing analysis: rec_ecndav')

    positions = to_nparray(positions).tolist()
    list_bag_ecn = []
    list_bag_dav = []
    list_bag_pij = []
    for index in range(len(positions)):
        logging.info('    Proceding analysis of structure index {:04d}'.format(
            index))
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


def symatoms(pd_df, bags, chemefeature=''):  # Funcao nao estavel
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


def findsc_wrap(inputs):  #It should be here
    """Wraper to multiprocessing findsc"""
    result = findsc(positions=inputs[0], atomic_radii=inputs[1],
                    adatom_radius=inputs[2], remove_is=inputs[3],
                    ssamples=inputs[4], writw_sp=inputs[5],
                    return_expositions=inputs[6],
                    print_surf_properties=inputs[7], sp_file=inputs[8])
    return result


def rec_findsc(positions, davraddii, davradius='dav', adatom_radius=1.1,  # Versao Nova/ Paralelo
               ssamples=1000, return_expositions=True,
               print_surf_properties=False, remove_is=True, procs=1):
    """It return the atom site surface(True)/core(Flase) for each atoms in
    for each structure pandas dataframe. See more in the function
    quandarium.analy.mols.findsurfatons.

    Parameters
    ----------
    adatom_radius: float (optional, default=1.1).
                   Radius of the dummy adatom, in angstroms.
    positions: Pandas.Series (optional, default='bag_positions')
               The name of the fuature (bag type) in pd_df with
               cartezian positions of the atoms.
    davraddii: str (optional, default='bag_dav')
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
    logging.info('    Initializing analysis: rec_findsc')

    inputs_list = []
    list_is_surface = []
    list_exposition = []
    for index, (poitionsi, davraddii) in enumerate(zip(positions, davraddii)):
        logging.info('    Proceding analysis of structure index {:04d}'.format(
            index))
        #print(type(poitionsi),poitionsi)
        positionsi = np.array(poitionsi)  # manter np.array e ativar bags
        if davradius == 'dav':
            atomic_radii = np.array(davraddii)/2  # manter np.array e ativar bags
        if davradius == 'radii':
            atomic_radii = bag2arr(davraddii)
        inputs_list.append([positionsi, atomic_radii, adatom_radius,
                            remove_is, ssamples, False, return_expositions,
                            print_surf_properties, "surface_points.xyz"])
    pool = mp.Pool(procs)
    s_time = time.time()
    size = len(inputs_list)
    result = pool.map_async(findsc_wrap, inputs_list, chunksize=1)

    while not result.ready():
        remaining = result._number_left
        print('Remaining: ', remaining)
        time.sleep(5.0)
    print('Finished')

    outputs = result.get()
    for index, _ in enumerate(outputs):
        list_is_surface.append(arr2bag(outputs[index][0]))
        list_exposition.append(arr2bag(outputs[index][1]))
    list_of_new_features_data = [list_is_surface, list_exposition]
    list_of_new_features_name = ['bag_issurf', 'bag_exposition']

    return list_of_new_features_name, list_of_new_features_data


def data_from_opsbags(pd_df, new_bag_name, bags, opsbags, opsinterbags='', kind='reg'):  # Versao Velha
    """It take a bag for operte over other bags.
    logicalop : np.logical_or, np.logical_and
    """

    print("Initializing bag_from_opsbags.")
    logging.info("Initializing bag_from_opsbags.")
    logging.info("new_bag_name: {}".format(new_bag_name))
    logging.info("bags: {}".format(bags))
    logging.info("opsbags: {}".format(opsbags))
    logging.info("opsinterbags: {}".format(opsinterbags))

    if opsinterbags == '':
        opsinterbags = [np.logical_and]*(len(bags) -1)

    new_bag_data = []
    for index in range(len(pd_df)):
        data = bag2arr(pd_df[bags[0]][index], dtype=str)
        operatedata = opsbags[0](data)
        for bag, bagop, interbagop in zip(bags[1:], opsbags[1:], opsinterbags):
            nextdata = bag2arr(pd_df[bag][index])
            operatednextdata = bagop(nextdata)
            operatedata = interbagop(operatedata, operatednextdata)
        if kind == 'bag':
            operatedata = arr2bag(operatedata)
        new_bag_data.append(operatedata)

    # criando um pd.DataFrame que tem todos os dados e nome dos mesmos
    new_df = pd.DataFrame(np.array([new_bag_data]).T,
                          columns=[new_bag_name])

    combined_df = pd.concat([pd_df, new_df], axis=1, sort=False)

    logcolumns('info', "New columns: ", new_df)

    return combined_df


def data_from_opsdatas(pd_df, new_bag_name, new_kind, kinds, bags, opsbags,  # Versao Velha
                       opsinterbags=''):
    """It take a bag for operte over other bags.
    logicalop : np.logical_or, np.logical_and
    """

    print("Initializing data_from_opsdatas.")
    logging.info("Initializing data_from_opsdatas.")
    logging.info("new_bag_name: {}".format(new_bag_name))
    logging.info("new_kind: {}".format(new_kind))
    logging.info("kinds: {}".format(kinds))
    logging.info("bags: {}".format(bags))
    logging.info("opsbags: {}".format(opsbags))
    logging.info("opsinterbags: {}".format(opsinterbags))

    if opsinterbags == '':
        opsinterbags = [np.logical_and]*(len(bags) -1)

    new_bag_data = []
    for index in range(len(pd_df)):
        logging.info('    Proceding analysis of structure index {:04d}'.format(
            index))
        if kinds[0] == 'bag':
            data = bag2arr(pd_df[bags[0]][index])
        else:
            data = pd_df[bags[0]][index]
        operatedata = opsbags[0](data)
        for bag, kind, bagop, interbagop in zip(bags[1:], kinds[1:], opsbags[1:], opsinterbags):
            if kind == 'bag':
                nextdata = bag2arr(pd_df[bag][index])
            else:
                nextdata = pd_df[bag][index]
            operatednextdata = bagop(nextdata)
            operatedata = interbagop(operatedata, operatednextdata)
        if new_kind == 'bag':
            operatedata = arr2bag(operatedata)
        new_bag_data.append(operatedata)

    # criando um pd.DataFrame que tem todos os dados e nome dos mesmos
    new_df = pd.DataFrame(np.array([new_bag_data]).T,
                          columns=[new_bag_name])

    combined_df = pd.concat([pd_df, new_df], axis=1, sort=False)

    logcolumns('info', "New columns: ", new_df)

    return combined_df


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
            for index in range(len(pd_df)):
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

    # if classesn1 == '':
    #    classesn1 = []
    #    for class1 in classes1:
    #        classesn1.append(class1.replace('bag_', ''))
    #    logging.info('automaticaly generated classesn1: ' + str(classesn1))

    # if classesn2 == '':
    #    classesn2 = []
    #    for class2 in classes2:
    #        classesn2.append(class2.replace('bag_', ''))
    #    logging.info('automaticaly generated classesn2: ' + str(classesn2))

    # ao longo da funcao serao adicionado os novos dados e nomes deles nessas
    # duas listas:
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


