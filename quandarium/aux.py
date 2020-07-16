"""This module present auxiliar function for cluster_analysis file"""

import sys
import ase.io
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

def to_nparray(data): # unir a de baixo
    """This functions recive a data and return a numpy array"""
    if isinstance(data, list):
        data = np.array(data)
    if isinstance(data, (pd.Series, pd.DataFrame)):
        data = np.array(data.values)
    if isinstance(data, np.ndarray):
        pass
    return data


def tonparray(*data, dropnan=True):
    """Converta data (pd series or non-flatten numpy array) to a flatten numpy
    array. Droping nans by default..."""
    if len(data) == 2:
        data1 = data[0]
        data2 = data[1]
        if isinstance(data1, pd.Series):
            data1 = data1.values.flatten()
        elif isinstance(data1, np.ndarray):
            data1 = data1.flatten()
        elif isinstance(data1, list):
            data1 = np.array(data1).flatten()
        else:
            print('ERROR: the type {} (for data1) is not suported in tonoparray.'.format(
                type(data1)))
        if isinstance(data2, pd.Series):
            data2 = data2.values.flatten()
        elif isinstance(data2, np.ndarray):
            data2 = data2.flatten()
        elif isinstance(data2, list):
            data2 = np.array(data2).flatten()
        else:
            print('ERROR: the type {} (for data2) is not suported in tonoparray.'.format(
                type(data2)))
        if dropnan:
            usefulldata = np.logical_and(np.isnan(data1) == False,
                                         np.isnan(data2) == False)
            data1, data2 = data1[usefulldata], data2[usefulldata]
        return data1, data2

    if len(data) == 1:
        data1 = data[0]
        if isinstance(data1, pd.core.series.Series):
            data1 = data1.values.flatten()
        elif isinstance(data1, np.ndarray):
            data1 = data1.flatten()
        elif isinstance(data1, list):
            data1 = np.array(data1).flatten()
        else:
            print('ERROR: the type {} is not suported in tonoparray.'.format(
                type(data1)))
        if dropnan:
            usefulldata = np.isnan(data1) == False
            data1 = data1[usefulldata]
    return data1


def to_list(data):
    """This functions recive a data and return a list"""
    if isinstance(data, (pd.Series, pd.DataFrame)):
        data = np.array(data.values).tolist()
    if isinstance(data, np.ndarray):
        data = data.tolist()
    return data


def checkmissingkeys(keys, dictlist, massange):
    """It return the missing keys of a dictionary or list"""
    missingkeyslist = []
    for key in keys:
        if key not in dictlist:
            missingkeyslist.append(key)
    if missingkeyslist:
        print("Error: " + massange + ': {}'.format(str(missingkeyslist)))
        sys.exit(1)


def translate_list(dictionary, list_to_be_translated):
    """For a given list with entries and a dictionary, it return a new list
    with the translated entries"""
    translated = []
    for i in list_to_be_translated:
        translated.append(dictionary[i])
    return translated


def comp_minmaxbond(atom1, atom2):
    """For the atoms 1 and 2, it return the max and min bond distances.
    Parameters
    ----------
    atom1, atom2: string.
                  A string with the atoms chemical elements symbol.
    Return
    ------
    minmax: list with two values.
            A list with the minimun and maximun distance bewteen two atoms to
            to consider they bounded.
    """

    conections = np.array([['H', 'H', 0.70, 1.19],
                           ['C', 'H', 0.90, 1.35],
                           ['C', 'C', 1.17, 1.51],
                           ['Fe', 'H', 1.2, 1.99],
                           ['Fe', 'C', 1.2, 2.15],
                           ['Fe', 'Fe', 2.17, 2.8],
                           ['Ni', 'H', 1.2, 1.98],
                           ['Ni', 'C', 1.2, 2.08],
                           ['Co', 'H', 1.2, 1.91],
                           ['Ni', 'Ni', 2.07, 2.66],
                           ['Co', 'C', 1.2, 2.20],
                           ['Co', 'Co', 2.05, 2.63],
                           ['Cu', 'Cu', 1.5, 2.76],
                           ['Cu', 'C', 1.2, 2.14],
                           ['Cu', 'H', 1.2, 1.98],
                           ['Zr', 'O', 1.1, 2.7],
                           ['Ce', 'O', 1.1, 2.7],
                           ['O', 'O', 1.1, 2.7],
                           ['Ce', 'Ce', 1.1, 2.7],
                           ['Zr', 'Zr', 1.1, 2.7],
                           ['Zr', 'Ce', 1.1, 2.7]])
    for info in conections:
        if ((atom1 in info[0]) and (atom2 in info[1])) or ((atom1 in info[1])
                                                           and
                                                           (atom2 in info[0])):
            result = info[2:]
    return result


def comp_gaussian(x, mu, sig):
    """Returns the valeues of a normalized gaussian (sig=sigma, mean=mu)
    over the values of x.
    Parameters
    ----------
    mu, sig: floats.
             Parameters of the gaussian function.
    x: numpy array (n,) shaped.
       Values to evaluate the normalized gaussian function.

    Retunr
    ------
    gaussian_of_x: np.array of lengh = len(x).
                   Values of the gaussian for the values in x.
    """
    return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)


def comp_pij_classed(pij, is_surface, cutoff):
    """It return the conectivity matrix considering only bounds between surface
    atoms.
    Parameters
    ----------
    pij: numpy array of floats, (n,n) shaped.
         Matrix of conectivity between atoms.
    is_surface: numpy array of boolean of len n.
                Array with boolean indicating if the i-th atom is belongs to
                the surface.
    cutoff: float > 0.
            Cutoff index to be considered as a bound.

    Return
    ------
    matrix_surface_bond: numpy array of boolean, (n,n) shaped.
                         Matrix indicating the surface atoms conecetivity.
    """
    ecn_bounded_ones = np.array(pij >= cutoff, dtype=int)
    is_surface_ones = np.array(is_surface, dtype=int)
    matrix_surface_bond = np.array((ecn_bounded_ones * is_surface_ones).T
                                   * is_surface_ones, dtype=bool)
    return matrix_surface_bond


def comp_pij_maxdamped(dij, ori, pij_max):
    """Compute all the conective index pij with a dumping"""
    exp_arg = (1 - (dij/(ori.reshape(-1, 1) + ori.reshape(1, -1)))**6
               - (dij/3.5)**4)
    return np.exp(exp_arg)*pij_max


def comp_aveabs(values):
    """It meansure the average of the absolute values
    Example
    -------
    >>> old = [1.21, 1.32, 1.23]
    >>> new = [1.20, 1.33, 1.24]
    >>> ave_abs(old-new)
    0.01
    """
    return np.average(np.abs(values))


def logistic(dij, ri, k):
    """Return the logistic function values:
    ri is the radius,
    k is the smmothness,
    dij is the atoms distance"""
    ri_sum_rj = ri.reshape(1, -1) + ri.reshape(-1, 1)
    return 1./(1. + np.exp(k*(dij - ri_sum_rj)))


def comp_rs(ori, dij, k, R, rcutp, w=[0.1, 0.60, 0.42]):  # pylint: disable w0102
    """Cost function optimized in ecn_rsopt function."""
    rcut = ori * rcutp

    ew = w[0]
    rRw = w[1]
    rdw = w[2]

    pij = logistic(dij, rcut, k)
    pijsum = np.sum(pij)

    # ecn costs
    ecn = np.sum(pij, axis=1)
    ecn_cost = ew * np.average(np.exp(-ecn))

    # r costs
    ori_sum_orj = ori.reshape([1, -1]) + ori.reshape([-1, 1])
    rd_cost = rdw * np.sum(((dij - ori_sum_orj)**2)*pij)/pijsum
    Ri_sum_Rj = R.reshape([1, -1]) + R.reshape([-1, 1])
    rR_cost = rRw * np.sum(((ori_sum_orj - Ri_sum_Rj)**2)*pij)/pijsum

    total = ecn_cost + rd_cost + rR_cost

    return total


def comp_roptl2(ori, dij, pij):
    """Compute the difference between ori + orj and the dij of the bonded
    atoms (pij) with a l2 norm."""
    ori_sum_orj = ori.reshape([1, -1]) + ori.reshape([-1, 1])
    return np.sum(((dij - ori_sum_orj)**2)*pij)


def cost_l2(ori, dij, pij):
    """L2 cost function for the difference between sum of two atoms radii and
    they distance, for bonded atoms.
    Parameters
    ----------
    ori: atoms radius, numpy.ndarray on lengh n.
         Atomic radii for each atom.
    dij, pij: numpy.ndarray (n,n) shaped.
              Atoms distances and index of conection.
    Result
    ------
    cost: float.
          The value of the costfunction.
    """

    if not isinstance(pij, np.ndarray):
        print("pij must be a (n,n) shaped np.array. Aborting...")
        sys.exit(1)

    if not isinstance(dij, np.ndarray):
        print("dij must be a (n,n) shaped np.array. Aborting...")
        sys.exit(1)

    if not isinstance(ori, np.ndarray):
        print("ori must be a (n,) shaped np.array. Aborting...")
        sys.exit(1)

    ri_sum_rj = ori.reshape([1, -1]) + ori.reshape([-1, 1])
    l2cost = np.sum(((dij - ri_sum_rj) * pij)**2)
    return l2cost


def changesymb(chemical_symbols, changes, condition=None):
    """It change the chemical symbols to new labels considering a condition.
    Parameters
    ----------
    chemical_symbols: np.array of strings, lengh n.
                      The chemical symbols for each atom.
    changes: dictionary of string to string.
             A dictionary of the old elements ralated with the new labels.
    condition: None or a numpy array of boolean, (n,) shaped.
               If None, the condition is allways true. If an array ware
               provided, the condition for each atom is in its ith value.
    Return
    ------
    new_labels: np.array of strings, lengh n.
                The new labels.
    """

    if condition is None:
        condition = np.ones(len(chemical_symbols), dtype=bool)
    new_labels = chemical_symbols.copy()
    symbols_to_change = list(changes.keys())
    for index, symbol in enumerate(chemical_symbols):
        if (symbol in symbols_to_change) and condition[index]:
            new_labels[index] = changes[chemical_symbols[index]]

    return new_labels


def fragstring(is_frag):
    """It return a string input for the get_charge script.

    Parameters
    ----------
    is_frag: np.array of boolean and lengh n.
             The array should present True for the atoms which belongs to the
             fragments.
    Return
    ------
    fragstring: a string with the fragment.
    """

    qtna = len(is_frag)
    atom_indexes = np.arange(0, qtna)
    is_frag_indexes = np.array(atom_indexes[is_frag], dtype=str)
    fragstr = '\'frag' + ','.join(is_frag_indexes) + '\''
    return fragstr


def RegRDS_set(sampling_distance, N):
    """Return a set of n dots in R3 (almost) regular distributed in the
    surface of a sphere of radius 'sampling_distance'.
    More deatiail of the implementation in the article "How to generate
    equidistributed points on the surface of a sphere" by Markus Deserno:
    https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf
    samplind_distance: a float or a int grater than zero.
    N: intiger grater than zero."""

    if not sampling_distance > 0:
        print("ERROR: sampling_distance must be higher than zero!")
        sys.exit(1)

    if (not isinstance(N, int)) or (N <= 0):
        print("ERROR: N must be an intiger grater than zero!")
        sys.exit(1)

    cart_coordinates = []
    r = 1
    Ncount = 0
    a = 4. * np.pi * r**2 / N
    d = np.sqrt(a)
    Mtheta = int(round(np.pi/d))
    dtheta = np.pi / Mtheta
    dphi = a / dtheta
    for m in range(0, Mtheta):
        theta = np.pi * (m + 0.5) / Mtheta
        Mphi = int(round(2*np.pi*np.sin(theta)/dphi))
        for n in range(0, Mphi):
            phi = 2 * np.pi * n / Mphi
            Ncount += 1
            y = sampling_distance * np.sin(theta) * np.cos(phi)
            x = sampling_distance * np.sin(theta) * np.sin(phi)
            z = sampling_distance * np.cos(theta)
            cart_coordinates.append([x, y, z])
    cart_coordinates = np.array(cart_coordinates)

    return cart_coordinates


def write_points_xyz(file_name, positions):
    """Write positions, a list or array of R3 points, in a xyz file file_named.
    Several softwares open xyz files, such as Avogadro and VESTA
    In the xyz file all the atoms are H.

    file_name: a string with the path of the xyz document which will be writed.
    positions: a list or numpy array with the atoms positions."""

    if not isinstance(file_name, str):
        print("file_name must be a string")
        sys.exit(1)

    for index, element in enumerate(positions):
        if len(element) != 3:
            print("Element {} of positions does not present three "
                  "elements.".format(str(index)))
    if not isinstance(positions, list):
        positions = np.array(positions)

    print("Writing points in the xyz file...")
    ase.io.write(file_name, ase.Atoms('H'+str(len(positions)),
                                      list(map(tuple, positions))))


def writing_molecule_xyz(file_name, positions, chemical_symbols):
    """Write xyz file from positions and chemical symbols in arrays.
    Several softwares open xyz files, such as Avogadro and VESTA

    Parameters
    ----------
    file_name: a string.
               The path of the xyz document which will be writed.
    positions: a list or numpy array (n,3) shaped.
               The cartezian atoms positions.

    Return
    ------
    Nothing
    """

    if not isinstance(file_name, str):
        print("file_name must be a string")
        sys.exit(1)
    for index, element in enumerate(positions):
        if len(element) != 3:
            print("Element " + str(index) + " of positions does not"
                  + "present three elements.")
    if not isinstance(positions, list):
        positions = np.array(positions)

    atoms = ase.Atoms(chemical_symbols, list(map(tuple, positions)))
    ase.io.write(file_name, atoms)


def dot_in_atom(dot, atom_position, atom_radius):
    """Verify if a dot in R3 is whitchin the atom of radius atom_radius located in
    atom_position.
    dot: np.array of 3 float elements.
    atom_position: np.array of 3 float elements.
    atom_radius: float grater than zero."""

    result = np.linalg.norm(dot - atom_position) < atom_radius
    return result


def large_surfaces_index(surface_dots_positions, eps):
    """Seach if there are more than one surfaces in surface_dots_positions, than
    return a np.array of booleans with True for index of atoms for the surfaces
    more points.
    surface_dots_positions: np.array with R3 dots.
    eps: minimun distance between different surfaces, should be a float grater
         than zero."""

    if (not isinstance(surface_dots_positions, np.ndarray)
            or (np.shape(surface_dots_positions)[1] != 3)
            or isinstance(surface_dots_positions[0][0], bool)):
        print("surface_dots_positions must be a (n,3) shaped np.array. Aborting...")
        sys.exit(1)

    if not eps > 0:
        print("eps must be large than zero.")
        sys.exit(1)

    db = DBSCAN(eps=eps, min_samples=1).fit_predict(surface_dots_positions)
    labels, quanity = np.unique(db, return_counts=True)
    if len(labels) > 1:
        print(str(len(labels)) + ' surfaces were found, of sizes: '
              + str(quanity).replace('[', '').replace(']', '')
              + '. The bigger will be selected!')
    result = db == labels[np.argmax(quanity)]

    return result
