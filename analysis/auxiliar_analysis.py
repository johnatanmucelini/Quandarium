import time
ti_header = time.time()
import sys
import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.stats import describe
from scipy.special import legendre
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from scipy.signal import find_peaks
import ase.io
import networkx as nx
import logging

logging.basicConfig(level=logging.INFO)


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


def bag2string(bag):
    """It print the data from bags.
    Parameters
    ----------
    bag: np.array of lengh n.
         It contains the values which will be converted in the string format.

    Return
    ------
    bagstring: string
               The values of bag as in the correct string format.
    """

    return '[' + ','.join(np.array(bag, dtype=str)) + ']'


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
    """Return a set of N R3 dots (almost) regular  distributed in the surface of
    a sphere of radius 'sampling_distance'.
    More deatiail of the implementation in the article "How to generate
    equidistributed points on the surface of a sphere" by Markus Deserno:
    https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf
    samplind_distance: a float or a int grater than zero.
    N: intiger grater than zero."""

    logging.debug("Initializing RegRDS_set function!")
    logging.debug("sampling_distance: " + str(sampling_distance))
    logging.debug("N: " + str(N))

    if not sampling_distance > 0:
        logging.error("sampling_distance must be higher than zero! Aborting...")
        sys.exit(1)

    if (type(N) != int) or (N <= 0):
        logging.error("N must be an intiger grater than zero! Aborting...")
        sys.exit(1)

    cart_coordinates=[]
    r = 1
    Ncount = 0
    a = 4. * np.pi * r**2 / N
    d = np.sqrt(a)
    Mtheta = int(round(np.pi/d))
    dtheta = np.pi / Mtheta
    dphi = a / dtheta
    logging.debug("Mtheta: " + str(Mtheta))
    for m in range(0, Mtheta ) :
        theta = np.pi *( m + 0.5 ) / Mtheta
        Mphi = int(round(2 *np.pi * np.sin(theta) / dphi ))
        logging.debug("Mtheta: " + str(Mphi))
        for n in range( 0 , Mphi ) :
            phi = 2* np.pi * n / Mphi
            Ncount += 1
            y = sampling_distance * np.sin(theta) * np.cos(phi)
            x = sampling_distance * np.sin(theta) * np.sin(phi)
            z = sampling_distance * np.cos(theta)
            cart_coordinates.append([x,y,z])
    cart_coordinates = np.array(cart_coordinates)
    logging.info("Final quanity of points in the radial grid: "
                 + str(len(cart_coordinates)))

    logging.debug("RegRDS_set function finished sucsessfuly!")
    return cart_coordinates


def writing_points_xyz(file_name, positions):
    """Write positions, a list or array of R3 points, in a xyz file file_named.
    Several softwares open xyz files, such as Avogadro and VESTA
    In the xyz file all the atoms are H.

    file_name: a string with the path of the xyz document which will be writed.
    positions: a list or numpy array with the atoms positions."""

    logging.debug("Initializing writing_points_xyz function!")
    logging.debug("file_name: " + str(file_name))
    logging.debug("positions: " + str(positions))

    if type(file_name) != str :
        logging.error("file_name must be a string")
        sys.exit(1)

    for index, element in enumerate(positions):
        if len(element) != 3:
            logging.error("Element " + str(index) + " of positions does not" \
                          "present three elements.")
    if type(positions) != list : positions = np.array(positions)

    logging.debug("Writing points in the xyz file...")
    ase.io.write( file_name , ase.Atoms('H'+str(len(positions)) , list(map(tuple,positions)) ))
    logging.debug("Finished.")

    logging.debug("writing_points_xyz function finished!")


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

    # logging.debug("Initializing writing_points_xyz function!")
    # logging.debug("file_name: " + str(file_name))
    # logging.debug("positions: " + str(positions))
    if not isinstance(file_name, str):
        print("file_name must be a string")
        sys.exit(1)
    for index, element in enumerate(positions):
        if len(element) != 3:
            print("Element " + str(index) + " of positions does not"
                  + "present three elements.")
    if not isinstance(positions, list):
        positions = np.array(positions)

    # logging.debug("Writing points in the xyz file...")
    atoms = ase.Atoms(chemical_symbols, list(map(tuple, positions)))
    ase.io.write(file_name, atoms)
    # logging.debug("Finished.")
    # logging.debug("writing_points_xyz function finished!")


def linspace_r3_vector ( vector_a , vector_b , Qtnsteps ) :
    """Return a np.array with elements that starting in vector_a and go to
    vector_b. The total quantity of dots is Qtnsteps.

    vector_a and vector_b: different np.array with floats in R3.
    Qtnsteps: intiger grater than zero."""

    logging.debug("Initializing linspace_r3_vector function!")

    if vector_a == vector_b :
        logging.error("vector_a is equal to vector_b. Aborting...")
        sys.exit(1)

    if (type(vector_a) != np.ndarray) or (len(vector_a) != 3) :
        logging.error("vector_a must be a np.array of len 3! Aborting..." )
        sys.exit(1)

    if (type(vector_b) != np.ndarray) or (len(vector_b) != 3) :
        logging.error("vector_b must be a np.array of len 3! Aborting..." )
        sys.exit(1)

    if (type(N) != int) or (N <= 0) :
        logging.error("N must be an intiger grater than zero! Aborting...")
        sys.exit(1)

    xvalues = np.linspace( vector_a[0] , vector_b[0] , Qtnsteps )
    yvalues = np.linspace( vector_a[1] , vector_b[1] , Qtnsteps )
    zvalues = np.linspace( vector_a[2] , vector_b[2] , Qtnsteps )
    final_array = np.array([xvalues, yvalues, zvalues]).T

    logging.debug("linspace_r3_vector finished sucsessfuly!")
    return final_array


def dot_in_atom( dot , atom_position , atom_radius ) :
    """Verify if a dot in R3 is whitchin the atom of radius atom_radius located in
    atom_position.
    dot: np.array of 3 float elements.
    atom_position: np.array of 3 float elements.
    atom_radius: float grater than zero."""

    logging.debug("Initializing dot_in_atom")

    result = np.linalg.norm( dot - atom_position ) < atom_radius
    logging.debug("result: " + str(result))

    logging.debug("dot_in_atom finished sucsessfuly!")
    return result


def large_surfaces_index(surface_dots_positions, eps):
    """Seach if there are more than one surfaces in surface_dots_positions, than
    return a np.array of booleans with True for index of atoms for the surfaces
    more points.
    surface_dots_positions: np.array with R3 dots.
    eps: minimun distance between different surfaces, should be a float grater
         than zero."""

    logging.debug("Initializing remove_pseudo_surfaces function!")

    if (isinstance(surface_dots_positions, np.ndarray)
            or (np.shape(surface_dots_positions)[1] != 3)
            or isinstance(surface_dots_positions[0][0], bool)):
        logging.error("surface_dots_positions must be a (n,3) shaped np.array.\
                  Aborting...")
        sys.exit(0)

    if not eps > 0:
        logging.error("eps must be large than zero.")
        sys.exit(1)

    db = DBSCAN(eps=eps, min_samples=1).fit_predict(surface_dots_positions)
    labels, quanity = np.unique(db, return_counts=True)
    if len(labels) > 1:
        logging.warning(str(len(labels)) + ' surfaces were found, of sizes: '
                        + str(quanity).replace('[', '').replace(']', '')
                        + '. The bigger will be selected!')
    result = db == labels[np.argmax(quanity)]

    logging.debug("remove_pseudo_surfaces finished sucsessfuly!")
    return result

def RandRDS_dot( sampling_distance ) :
    """Randon R3 dot in the surface of a sphere of radius sampling distance.
    See more details in: http://mathworld.wolfram.com/SpherePointPicking.html
    samplind_distance: a float or a int grater than zero."""

    logging.debug("Initializing the RandRDS_dot function.")
    logging.debug("sampling_distance: " + str(sampling_distance))

    if not sampling_distance > 0 :
        logging.error("sampling_distance must be higher than zero! Aborting...")
        sys.exit(0)

    u , v = np.random.random(2)
    logging.debug("u, v: " + str(u) + ", " + str(v))

    theta = 2 * np.pi * v
    phi = np.arccos(2*u-1)
    logging.debug("theta, phi: " + str(theta) + ", " + str(phi))

    x = sampling_distance * np.cos(theta) * np.sin(phi)
    y = sampling_distance * np.sin(theta) * np.sin(phi)
    z = sampling_distance * np.cos(phi)
    Dot = np.array([x,y,z])
    logging.debug("Dot: " + str(Dot))

    logging.debug("RandRDS_dot function finished sucsessfuly!")
    return Dot


def RandRDS_set( sampling_distance , N ) :
    """Return a set of N randon R3 dots in the surface of a sphere of radius
    sampling distance.
    See more details in: http://mathworld.wolfram.com/SpherePointPicking.html
    samplind_distance: a float or a int grater than zero.
    N: intiger grater than zero."""

    logging.debug("Initializing the RandRDS_set function.")
    logging.debug("sampling_distance: " + str(sampling_distance) )
    logging.debug("N: " + str(N))

    if not sampling_distance > 0 :
        logging.error("sampling_distance must be higher than zero! Aborting...")
        sys.exit(1)

    if (type(N) != int) or (N <= 0) :
        logging.error("N must be an intiger grater than zero! Aborting...")
        sys.exit(1)

    cart_coordinates=[]
    for i in range(0, N):
        cart_coordinates.append( RandRDS_dot(sampling_distance) )
    cart_coordinates = np.array(cart_coordinates)

    logging.debug("RandRDS_set function finished sucsessfuly!")
    return cart_coordinates
