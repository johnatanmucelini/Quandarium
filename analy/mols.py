""" An algorithm to analyse the structure and geometry of cluster of atoms."""

import sys
import itertools
import time
import logging
import ase.io
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.stats import describe
from scipy.special import legendre
from scipy.spatial.distance import cdist
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN
from quandarium.analy.aux import RegRDS_set
from quandarium.analy.aux import large_surfaces_index
from quandarium.analy.aux import write_points_xyz
from quandarium.analy.aux import comp_aveabs
from quandarium.analy.aux import comp_roptl2
from quandarium.analy.aux import comp_gaussian
from quandarium.analy.aux import comp_pij_maxdamped
from quandarium.analy.aux import comp_pij_classed
from quandarium.analy.aux import comp_minmaxbond
from quandarium.analy.aux import comp_rs
from quandarium.analy.aux import logistic

logging.basicConfig(filename='/home/johnatan/quandarium_module.log',
                    level=logging.INFO)
logging.info('The logging level is INFO')
np.random.seed(1234)

def avradius(positions, raddii, useradius=True):
    """Return the average radius of some molecule.
    Parameters
    ----------
    positions: numpy array of floats (n,3) shaped.
               Cartezian positions of the atoms, in angstroms.
    atomic_radii: numpy array of floats (n,) shaped.
                  Radius of the atoms, in the same order which they appear in
                  positions, in angstroms.

    Returns
    -------
    average_radius: numpy float
    """
    if len(positions) > 1:
        positions = positions - np.average(positions, axis=0)
        dists = cdist(positions, positions)
        distargmax = np.unravel_index(np.argmax(dists), dists.shape)
        part1 = dists[distargmax[0], distargmax[1]]
        if useradius:
            part1 += raddii[distargmax[0]] + raddii[distargmax[1]]
            distcenter = np.linalg.norm(positions, axis=0) + raddii
            part2 = max(distcenter)
        else:
            distcenter = np.linalg.norm(positions, axis=0)
            part2 = max(distcenter)
        average_radius = part1/2. + part2/2.
    else:
        average_radius = np.nan
    return average_radius


def ecndav_rsopt(positions, cheme, kinfo, Rinfo, roundpijtoecn=False):
    """Return an effective coordination number (ecn), the optimized atomic
    radius (ori), the bond distance average (dav), and the conective index
    matrix pij. The radius are optimized...

    Parameters
    ----------
    positions: numpy.array, (3,n) shaped.
               The cartezian positions of the atoms.
    cheme : numpy.array, (n,) shaped.
            the chemical symbols of each element.
    kinfo : float, np.array, or dictionary.
            Coordination activation factor, a float will give the same factor k
            for each possible coordination. A np.array (n,n) shaped, where n is
            the quantity of atoms, will be consider the each entry as the k for
            each pair of possible coordination. A dictionary will construct
            each k factor as dict[cheme[i]] plus dict[cheme[j]].
    Rinfo : np.array or a dict.
            The atomic tabeled radius. If a dict, each element radius will be
            consider as dict[cheme[i]].
    roundpijtoecn: If true the pij is rounded to calculate the ecn.

    Returns
    -------
    ecn, dav, ori : numpy.array, (n,) shaped.
                   They contain the calculated ecn, optimized radius, and dav
                   for each atom.
    pij: numpy.array, (n,n) shaped.
         The index of connectivity between pairs of atoms.
    """

    qtna = len(positions)
    positions = positions - np.average(positions, axis=0)
    dij = cdist(positions, positions) + 100 * np.eye(qtna)

    if isinstance(kinfo, float):
        k = np.ones([qtna, qtna]) * kinfo
    elif isinstance(kinfo, dict):
        k = np.array([[kinfo[e1] + kinfo[e2] for e1 in cheme] for e2 in cheme])
    else:
        k = kinfo

    if isinstance(Rinfo, dict):
        R = np.array([kinfo[e1] for e1 in cheme])
    else:
        R = Rinfo

    rs = np.zeros(qtna*2)
    rcut_min = np.min(dij, axis=1) * 0.30
    rcut_max = 3*rcut_min
    bounds = []
    # rcut
    for ind, ce in enumerate(cheme):
        rs[ind] = R[ind]
        bounds.append((rcut_min[ind], rcut_max[ind]))
    # ori
    for ind, ce in enumerate(cheme):
        rs[ind + len(positions)] = R[ind] * 1.9
        bounds.append((rcut_min[ind], rcut_max[ind]))
    bounds = tuple(map(tuple, bounds))
    # print('{:<3}  {:^3}  {:^3}  {:^3}'.format('ce','r_min','r_max','rcut'))
    # for ind in range(len(positions)):
    #     print('{}   {}   {}   {}'.format(cheme[ind], bounds[ind][0],
    #                                      bounds[ind][1], rs[ind]))
    # print(bounds)

    rs_opt = optimize.minimize(comp_rs, rs, args=(dij, k, R),
                               bounds=bounds,
                               method="L-BFGS-B", tol=1.E-6,
                               options={"maxiter": 250, "disp": False})

    # print('av_ecn,  total,     rdij,         rR,      rrcut,       ecn,     '
    #       'ecndump')
    # print('\n\n' + str(rs_opt.message) + '\n' + str(rs_opt.success) + '\n\n')
    if not rs_opt.success:
        logging.warning('Otimization do not converg.')

    rcut = rs_opt.x[:qtna]
    ori = rs_opt.x[qtna:]
    pij = logistic(dij, rcut, k)
    ecn = np.sum(pij, axis=1)
    ecn_aux = ecn < 1.
    print(ecn[ecn_aux == False], np.ones_like(ecn)[ecn_aux])
    ecn_aux2 = ecn
    ecn_aux2[ecn_aux] = 1. 
    dav = np.sum(dij*pij, axis=1) / ecn_aux2
    if roundpijtoecn:
        ecn = np.sum(np.round(pij), axis=1)

    return ecn, dav, ori, pij


def ecndav_ropt(positions, chemical_symbols, plot_name='',
                print_convergence=True, roundpijtoecn=False):
    """Return the effective coordination number (ecn), the optimized atomic
    radius (ori), the bond distance average (dav), and the conective index
    matrix pij. The radius are optimized...

    Parameters
    ----------
    positions: numpy.array, (3,n) shaped.
               The cartezian positions of the atoms.
    chemical_symbols : numpy.array, (n,) shaped.
                       the chemical symbols of each atom.
    criteria: float greater than zero.
              Creteria for connectivity.
    print_convergence: boolean, (optional, default=True).
                       It treu, the convergency will be printed.
    plot_name: string (optional, defauld='').
               It a string were provided, a plot of the dij as function of ori
               + orj will be saved in for each atom i, with name
               plot_name_i.png.
    roundpijtoecn: If true the pij is rounded to calculate the ecn.

    Returns
    -------
    ecn, ori, dav: numpy.array, (n,) shaped.
                   They contain the calculated ecn, optimized radius, and dav
                   for each atom.
    Pij: numpy.array, (n,n) shaped.
         The index of connectivity between pairs of atoms.
    """


    if print_convergence:
        logging.info('Initializing ECN-Ropt analysis!')
        print("Initializing ECN-Ropt analysis!")
    else:
        logging.debug('Initializing ECN-Ropt analysis!')

    if not isinstance(positions, np.ndarray) or (np.shape(positions)[1] != 3):
        print("positions must be a (n,3) shaped numpy.ndarray! Aborting...")
        sys.exit(1)

    if not isinstance(chemical_symbols, np.ndarray):
        print("chemical_symbols must be a numpy.ndarray of strings.")
        sys.exit(1)

    qtna = len(positions)
    dij = cdist(positions, positions) + 100 * np.eye(len(positions))
    dav = np.max(cdist(positions, positions), axis=0)
    ori = dav / 2.
    ori_pre = np.zeros_like(ori)
    ecn_pre = np.zeros_like(dij)
    ecn = np.zeros_like(dij)

    # dij_max = np.zeros([qtna,qtna])
    # for ia1, a1 in enumerate(chemical_symbols):
    #     for ia2, a2 in enumerate(chemical_symbols):
    #         dij_max[ia1,ia2] = comp_minmaxbond(a1, a2)[1]
    #         print(a1, a2, dij_max[ia1,ia2], dij[ia1,ia2])
    dij_max = np.array([[comp_minmaxbond(a1, a2)[1] for a1 in chemical_symbols]
                        for a2 in chemical_symbols], dtype=float)
    pij_max = dij < dij_max
    step = 0
    logging.debug('     Delta sum_i(abs(r_i))/N     Delta sum_i(abs(ECN_i))/N"')
    if print_convergence:
        print("     Delta sum_i(abs(r_i))/N     Delta sum_i(abs(ECN_i))/N")
    while (np.sum(np.abs(ori_pre - ori)) / len(ori) > 10E-8) or (step < 2):
        if step > 0:
            ori_pre = ori * 1.
            ecn_pre = ecn * 1.
        pij = comp_pij_maxdamped(dij, ori, pij_max)
        results = optimize.minimize(comp_roptl2, ori, args=(dij, pij),
                                    bounds=((0.5, 1.9),)*len(chemical_symbols),
                                    method="L-BFGS-B", tol=1.E-7,
                                    options={"maxiter": 50, "disp": False})
        ori = results.x
        ecn = np.sum(pij, axis=1)
        parameter1 = comp_aveabs(ori_pre - ori)
        parameter2 = comp_aveabs(ecn - ecn_pre)
        logging.debug('    {}    {}'.format(parameter1, parameter2))
        if print_convergence:
            print('   ', parameter1, parameter2)
        step += 1
    if not results.success:
        logging.error('    Final r optimiation failed! see the massange: '
                      '{:s}'.format(results.message))
    else:
        logging.debug('    Final r optimization successfully finished!')

    logging.debug('    Self consistence achived!')

    if roundpijtoecn:
        logging.debug('    Rounding pij and calculating ECN')
        ecn = np.sum(np.round(pij), axis=1)

    # dav
    ori_sum_orj = ori.reshape([1, -1]) + ori.reshape([-1, 1])
    # logging.info(str(chemical_symbols[np.argmin(np.sum(pij, axis=1))]) + ' '
    #              + str(min(np.sum(pij, axis=1))) + ' '
    #              + str(np.min(dij, axis=1)) + ' ' + str(np.sum(pij, axis=1)))
    dav = np.sum((ori_sum_orj)*pij, axis=1) / np.sum(pij, axis=1)

    if plot_name:
        bins = 400
        sig = 0.05
        xvalues = np.linspace(0.5, 2, bins)
        rd_per_atom = np.zeros([len(chemical_symbols), bins])
        # ecn_int = np.zeros(len(chemical_symbols))
        for i in range(qtna):
            plt.close("all")
            for j in range(qtna):
                rd_per_atom[i] += comp_gaussian(xvalues,
                                                dij[i, j]/(ori[i] + ori[j]),
                                                sig)
            peaks, _ = find_peaks((rd_per_atom[i]+1)**-1, height=0)
            plt.plot(rd_per_atom[i])
            plt.plot(peaks, rd_per_atom[i][peaks], "x")
            plt.plot(np.zeros_like(rd_per_atom[i]), "--", color="gray")
            # ecn_int[i] = np.trapz(rd_per_atom[i,0:peaks[0]],
            #                       x=xvalues[0:peaks[0]])
            plt.savefig(plot_name + '_' + str(i) + '.png')

    if print_convergence:
        logging.info('    Analysis concluded!')
    else:
        logging.debug('    Analysis concluded!')

    return ecn, dav, ori, pij


def ecndav(positions, print_convergence=True):
    """Return the effective coordination number (ecn) and the average bound
    distance and the conective index matrix Pij.

    Parameters
    ----------
    positions: numpy.array, (3,n) shaped.
               The cartezian positions of the atoms.
    print_convergence: boolean, (optional, default=True).
                       It treu, the convergency will be printed.
    Returns
    -------
    ecn, dav: numpy.array, (n,) shaped.
              They contain the calculated ecn and dav for each atom.
    pij: numpy.array, (n,n) shaped.
         The index of connectivity between pairs of atoms.
    """

    if (not isinstance(positions, np.ndarray)) or np.shape(positions)[1] != 3:
        print("positions must be a (n,3) shaped numpy.ndarray! Aborting...")
        sys.exit(1)

    if print_convergence:
        print("ECN analysis:")
    qtna = len(positions)
    dij = cdist(positions, positions) + 100*np.eye(len(positions))
    dav = np.max(cdist(positions, positions), axis=0)
    dav_pre = np.zeros_like(dav)
    ecn_pre = np.zeros_like(dij)
    ecn = np.zeros_like(dij)

    step = 0
    logging.debug('    Delta sum_i(abs(dav_i))/N    Delta sum_i(abs(ECN_i))/N')
    if print_convergence:
        print("     Delta sum_i(abs(dav_i))/N     Delta sum_i(abs(ECN_i))/N")
    while (np.sum(np.abs(dav_pre - dav))/len(dav) > 10E-8) or (step < 2):
        if step > 0:
            dav_pre = dav * 1.
            ecn_pre = ecn * 1.
        pij = np.exp(1 - (2*dij/(dav.reshape(-1, 1) + dav.reshape(1, -1)))**6)
        ecn = np.sum(pij, axis=1)
        dav = np.sum(pij * dij, axis=1) / np.sum(pij, axis=1)
        ecn = np.sum(pij, axis=1)
        if print_convergence:
            print('   ' + str(np.sum(np.abs(dav_pre - dav)) / qtna)
                  + '  ' + str(np.sum(np.abs(ecn - ecn_pre)) / qtna))
        step += 1

    if print_convergence:
        print("Converged")
        logging.debug('    Self consistence achived!')

    return ecn, dav, pij


def findsc(positions, atomic_radii, adatom_radius, remove_is=True,
           ssamples=1000, writw_sp=True, return_expositions=True,
           print_surf_properties=False, sp_file="surface_points.xyz"):
    """This algorithm classify atoms in surface and core atoms employing the
    concept of atoms as ridge spheres. Then the surface atoms are the ones that
    could be touched by an fictitious adatom that approach the cluster, while
    the core atoms are the remaining atoms.
    See more of my algorithms im GitHub page Johnatan.mucelini.
    Articles which employed thi analysis: Mendes P. XXX
    .

    Parameters
    ----------
    positions: numpy array of floats (n,3) shaped.
               Cartezian positions of the atoms, in angstroms.
    atomic_radii: numpy array of floats (n,) shaped.
                  Radius of the atoms, in the same order which they appear in
                  positions, in angstroms.
    adatom_radius: float (optional, default=1.1).
                   Radius of the dummy adatom, in angstroms.
    ssampling: intiger (optional, default=1000).
               Quantity of samplings over the touched sphere surface of each
               atom.
    write_sp: boolean (optional, default=True).
              Define if the xyz positions of the surface points will
              be writed in a xyz file (surface_points.xyz).
    sp_file : string (optional, default='surface_points.xyz').
              The name of the xyz file to write the surface points positions,
              in angstroms, if write_sp == True.

    Return
    ------
    surface_exposition: numpy array of floats (n,).
                        The percentual of surface exposition of each atom.

    Example
    ------
    >>> ...
    """

    # Centralizing atoms positions:
    positions = positions - np.average(positions, axis=0)
    touch_radii = atomic_radii + adatom_radius
    qtna = len(positions)

    dots_try = RegRDS_set(adatom_radius, ssamples)
    rssamples = len(dots_try)
    max_dots = rssamples * qtna
    if print_surf_properties:
        print('Quantity of dots per atom: ' + str(len(dots_try)))
        print('Quantity of investigated dots: ' + str(max_dots))

    dots = positions[0] + RegRDS_set(touch_radii[0] + 0.001, ssamples)
    dot_origin = [[0] * len(dots_try)]
    for atom_id in range(1, qtna):
        dots = np.append(dots, positions[atom_id] + RegRDS_set(
            touch_radii[atom_id] + 0.001, ssamples), axis=0)
        dot_origin.append([atom_id] * len(dots_try))
    dot_origin = np.array(dot_origin).flatten()

    # removing dots inside other touch sphere
    dots_atoms_distance = cdist(positions, dots)
    atomos_radii_projected = np.array(
        [touch_radii]*len(dots)
        ).reshape(len(dots), qtna).T
    surface_dots = np.sum(
        dots_atoms_distance < atomos_radii_projected,
        axis=0
        ) == 0
    dots_surface = dots[surface_dots]

    # removing internal surfaces
    if remove_is:
        if print_surf_properties:
            print("removing_internal_surface")
        dots_for_find_eps = RegRDS_set(max(touch_radii), ssamples)
        dotdot_distances = cdist(dots_for_find_eps, dots_for_find_eps)
        min_dotdot_dist = np.min(dotdot_distances + np.eye(rssamples) * 10,
                                 axis=0)
        eps = 2.1 * np.max(min_dotdot_dist)
        external_surface_dots = large_surfaces_index(dots_surface, eps)
        dots_surface = dots_surface[external_surface_dots]
        surface_dot_origin = dot_origin[surface_dots][external_surface_dots]
    else:
        surface_dot_origin = dot_origin[surface_dots]
    surface_atoms, dots_per_atom = np.unique(surface_dot_origin,
                                             return_counts=True)

    # Getting exposition
    atoms_area_per_dot = (4 * np.pi * atomic_radii**2)/(1.*rssamples)
    exposition = np.zeros(qtna)
    is_surface = np.zeros(qtna, dtype=bool)
    incidence = np.zeros(qtna)
    # dots_per_atom = np.zeros(qtna)
    for atom_id, atom_incidence in zip(surface_atoms, dots_per_atom):
        if print_surf_properties:
            print(' found surface atom: ' + str(atom_id))
        is_surface[atom_id] = True
        incidence[atom_id] = atom_incidence
        exposition[atom_id] = atom_incidence * atoms_area_per_dot[atom_id]

    if writw_sp:
        write_points_xyz(sp_file, dots_surface)

    if print_surf_properties:
        centered_dots_surface = dots_surface - np.average(dots_surface, axis=0)
        origin = np.array([[0., 0., 0.]])
        dots_origin_dist = cdist(origin, centered_dots_surface).flatten()
        print("Surface points description: " + str(describe(dots_origin_dist)))
        print("reg_surface area: " + str(sum(exposition)))

    if not return_expositions:
        return is_surface
    if return_expositions:
        return is_surface, exposition


def bond_connective(positions, chemical_symbols):
    printt( pl , 1 , "\n\nBound counting analysis...")

    def translate_list (dictionary , list_to_be_translated ) :
        translated=[]
        for i in list_to_be_translated:
            translated.append( dictionary[i] )
        return translated

    quantity_of_atoms = Qtna
    distance = cdist( positions , positions )

    atoms = []
    atoms_types = []
    for i in range( 0 , quantity_of_atoms ) :
        atoms_types.append( chemical_symbols[i] )
        atoms.append( chemical_symbols[i] + "_" + str(i) )

    # bonds
    bonded_ecn_condition = 1.
    print( '    Bonded condition: ECN_{ij} ==' , bonded_ecn_condition )
    bonded = pij == 1.
    ##

    ##
    # processing
    ti_processing = time.time()
    G = nx.Graph()
    for i in range( 0 , quantity_of_atoms ):
        G.add_node( atoms[i] , atom_type=atoms_types[i] )
    for i in range(0 , quantity_of_atoms ) :
        for j in range( i+1 , quantity_of_atoms ) :
            if bonded[i,j]:
                G.add_edge( atoms[i], atoms[j] , distance=distance[i,j])

    recorrent_bonds = []
    nonrecorrent_bonds = []
    atom_type = nx.get_node_attributes(G , 'atom_type' )
    #types_of_atoms = np.unique(atoms_types)
    types_of_atoms = [ 'Zr' , 'Ce' , 'O' ]
    atoms_dummy_class = np.zeros( [len(atoms),len(types_of_atoms)] , dtype=int )
    first_degree_bonds_in_lines = np.zeros( [len(atoms),len(types_of_atoms)] , dtype=int )
    second_degree_bonds_in_lines = np.zeros( [len(atoms),len(types_of_atoms),len(types_of_atoms)] , dtype=int )
    third_degree_bonds_in_lines = np.zeros( [len(atoms),len(types_of_atoms),len(types_of_atoms),len(types_of_atoms)] , dtype=int )
    first_degree_bonds_in_cicles = np.zeros( [len(atoms),len(types_of_atoms)] , dtype=int )
    second_degree_bonds_in_cicles = np.zeros( [len(atoms),len(types_of_atoms),len(types_of_atoms)] , dtype=int )
    third_degree_bonds_in_cicles = np.zeros( [len(atoms),len(types_of_atoms),len(types_of_atoms),len(types_of_atoms)] , dtype=int )
    first_degree_bonds_in_backs = np.zeros( [len(atoms),len(types_of_atoms)] , dtype=int )
    second_degree_bonds_in_backs = np.zeros( [len(atoms),len(types_of_atoms),len(types_of_atoms)] , dtype=int )
    third_degree_bonds_in_backs = np.zeros( [len(atoms),len(types_of_atoms),len(types_of_atoms),len(types_of_atoms)] , dtype=int )

    for atom_index in range(0 , len(atoms)) :
        recorrent_bonds.append( [] )
        nonrecorrent_bonds.append( [] )
        current_atom = atoms[atom_index]
        for atom_type_index in range(0 , len(types_of_atoms)) :
            if atom_type[current_atom] == types_of_atoms[atom_type_index] :
                atoms_dummy_class[ atom_index , atom_type_index ] = 1
        neighbors1 = list(G.neighbors(current_atom))
        neighbors1_types = np.array(translate_list( atom_type, neighbors1 ))
        for neighborn1 in neighbors1 :
            for neighborn1_atom_type_index in range(0 , len(types_of_atoms)) :
                first_degree_bonds_in_lines[ atom_index , neighborn1_atom_type_index ] += int( atom_type[neighborn1]  == types_of_atoms[neighborn1_atom_type_index] )
                first_degree_bonds_in_cicles[ atom_index , neighborn1_atom_type_index ] += int( atom_type[neighborn1]  == types_of_atoms[neighborn1_atom_type_index] )
                first_degree_bonds_in_backs[ atom_index , neighborn1_atom_type_index ] += int( atom_type[neighborn1]  == types_of_atoms[neighborn1_atom_type_index] )
            neighbors2 = list(G.neighbors(neighborn1))
            neighbors2_types = np.array(translate_list( atom_type, neighbors2 ))
            for neighborn2 in neighbors2 :
                for neighborn1_atom_type_index in range(0 , len(types_of_atoms)) :
                    for neighborn2_atom_type_index in range(0 , len(types_of_atoms)) :
                        second_degree_bonds_in_backs[ atom_index , neighborn1_atom_type_index , neighborn2_atom_type_index ] += int( (atom_type[neighborn1] == types_of_atoms[neighborn1_atom_type_index] )*( atom_type[neighborn2] == types_of_atoms[neighborn2_atom_type_index]) )
                        if current_atom != neighborn2 :
                            second_degree_bonds_in_cicles[ atom_index , neighborn1_atom_type_index , neighborn2_atom_type_index ] += int( (atom_type[neighborn1] == types_of_atoms[neighborn1_atom_type_index] )*( atom_type[neighborn2] == types_of_atoms[neighborn2_atom_type_index]) )
                        if len(set([current_atom , neighborn1, neighborn2])) == 3 :
                            second_degree_bonds_in_lines[ atom_index , neighborn1_atom_type_index , neighborn2_atom_type_index ] += int( (atom_type[neighborn1] == types_of_atoms[neighborn1_atom_type_index] )*( atom_type[neighborn2] == types_of_atoms[neighborn2_atom_type_index]) )
                neighbors3 = list(G.neighbors(neighborn2))
                neighbors3_types = np.array(translate_list( atom_type, neighbors3 ))
                for neighborn3 in neighbors3 :
                    for neighborn1_atom_type_index in range(0 , len(types_of_atoms)) :
                        for neighborn2_atom_type_index in range(0 , len(types_of_atoms)) :
                            for neighborn3_atom_type_index in range(0 , len(types_of_atoms)) :
                                third_degree_bonds_in_backs[ atom_index , neighborn1_atom_type_index , neighborn2_atom_type_index , neighborn3_atom_type_index ] += int( (atom_type[neighborn1] == types_of_atoms[neighborn1_atom_type_index] )*( atom_type[neighborn2] == types_of_atoms[neighborn2_atom_type_index])*( atom_type[neighborn3] == types_of_atoms[neighborn3_atom_type_index]) )
                                if current_atom != neighborn2 and neighborn1 != neighborn3 :
                                    third_degree_bonds_in_cicles[ atom_index , neighborn1_atom_type_index , neighborn2_atom_type_index , neighborn3_atom_type_index ] += int( (atom_type[neighborn1] == types_of_atoms[neighborn1_atom_type_index] )*( atom_type[neighborn2] == types_of_atoms[neighborn2_atom_type_index])*( atom_type[neighborn3] == types_of_atoms[neighborn3_atom_type_index]) )
                                if len(set([current_atom , neighborn1, neighborn2, neighborn3 ])) == 4 :
                                    third_degree_bonds_in_lines[ atom_index , neighborn1_atom_type_index , neighborn2_atom_type_index , neighborn3_atom_type_index ] += int( (atom_type[neighborn1] == types_of_atoms[neighborn1_atom_type_index] )*( atom_type[neighborn2] == types_of_atoms[neighborn2_atom_type_index])*( atom_type[neighborn3] == types_of_atoms[neighborn3_atom_type_index]) )
                    recorrent_bonds[atom_index].append( [ neighborn1, neighborn2 , neighborn3] )
                    if atoms[atom_index] != neighborn2 and neighborn1 != neighborn3:
                        nonrecorrent_bonds[atom_index].append( [ neighborn1 , neighborn2 , neighborn3] )


    atom_types_to_be_printed = types_of_atoms.copy()

    print( '    Counting type:', count)
    if count == 'lines' :
        first_degree_bonds = first_degree_bonds_in_lines
        second_degree_bonds = second_degree_bonds_in_lines
        third_degree_bonds = third_degree_bonds_in_lines
    if count == 'cicles' :
        first_degree_bonds = first_degree_bonds_in_cicles
        second_degree_bonds = second_degree_bonds_in_cicles
        third_degree_bonds = third_degree_bonds_in_cicles
    if count == 'backs' :
        first_degree_bonds = first_degree_bonds_in_backs
        second_degree_bonds = second_degree_bonds_in_backs
        third_degree_bonds = third_degree_bonds_in_backs

    string0=[]
    string1=[]
    string2=[]
    string3=[]
    for atom_index in range(0 , len(atoms)) :
        string0.append( ','.join(np.array( atoms_dummy_class[atom_index]             , dtype=str )) )
        string1.append( ','.join(np.array( first_degree_bonds[atom_index].flatten()  , dtype=str )) )
        string2.append( ','.join(np.array( second_degree_bonds[atom_index].flatten() , dtype=str )) )
        string3.append( ','.join(np.array( third_degree_bonds[atom_index].flatten()  , dtype=str )) )
    string0 = np.array(string0)
    string1 = np.array(string1)
    string2 = np.array(string2)
    string3 = np.array(string3)

    # header
    header0=','.join(np.array(types_of_atoms,dtype=str))
    bonds_string=[]
    for i in itertools.product(',',types_of_atoms):
        bonds_string.append( '-'.join(i) )
    header1= ''.join(bonds_string)[1:]
    bonds_string=[]
    for i in itertools.product(',',types_of_atoms,types_of_atoms):
        bonds_string.append( '-'.join(i) )
    header2= ''.join(bonds_string)[1:]
    bonds_string=[]
    for i in itertools.product(',',types_of_atoms,types_of_atoms,types_of_atoms):
        bonds_string.append( '-'.join(i) )
    header3= ''.join(bonds_string)[1:]

    sep=np.array( [',',',',','] )

    #if degree == 1 :
    #    header2 = ''
    #    header3 = ''
    #    string2[:] = ''
    #    string3[:] = ''
    #    sep[1:] = ''
    #if degree == 2 :
    #    header1 = ''
    #    header3 = ''
    #    string1[:] = ''
    #    string3[:] = ''
    #    sep[1:] = ''
    #if degree == 3 :
    #header2 = ''
    #header1 = ''
    #string2[:] = ''
    #string1[:] = ''
    #sep[1:] = ''

    header_string = '[' + header0 + sep[0] + header1 + sep[1] + header2 + sep[2] + header3 + ']'

    print( '    bag_conections_information:' + header_string )
    final=','
    initial='    bag_of_bag_of_conections: ['
    for atom_index in range(0 , quantity_of_atoms) :
        if atoms_types[atom_index] in atom_types_to_be_printed :
           if atom_index == quantity_of_atoms - 1 : final=']\n'
           if atom_index > 0 : initial=''
           print( initial + '[' + string0[atom_index] + sep[0] + string1[atom_index] + sep[1] + string2[atom_index] + sep[2] + string3[atom_index] + ']' , end =final )










########################
########################

def planes_of_atoms():
    #### surface topology
    printt( pl , 1 , "\n\nSurface atoms topology analysis..." )

    # finding surface_bounded_axis
    print( 'Bounded criteria: ECN_{ij} < 1.')
    surface_bounded = comp_pij_classed( pij , is_surface , 1. ) # important parameter
    printt( pl , 1 , "    reg_bound_in_surface: " + str(np.sum(surface_bounded)/2.))

    # finding initial planes
    print('    Finding initial planes...')
    random_params=np.random.random(4)
    planes = []
    atoms_in_planes = []
    planes_scores = []
    for atom_1_index in range(0 , len(is_surface)) :
        for atom_2_index in range(atom_1_index+1,len(is_surface)) :
            for atom_3_index in range(atom_2_index+1,len(is_surface)) :
                if surface_bounded[atom_1_index, atom_2_index] and surface_bounded[atom_2_index, atom_3_index] and surface_bounded[atom_3_index, atom_1_index] :
                    plane = best_plane( random_params, np.array([positions[atom_1_index],positions[atom_2_index],positions[atom_3_index]]) , summed_sqd_errors )
                    planes.append( plane )
                    atoms_in_planes.append( [atom_1_index , atom_2_index , atom_3_index] )
                    planes_scores.append( plane_score_function(plane , positions[[atom_1_index , atom_2_index , atom_3_index]]) )
    print('    Quantity of intial planes:' , len(planes) )

    if len(planes) > 0 :
        # Seaching atoms with are in the plane
        print('    Finding atoms for each of initial planes...')
        for plane_index , plane in enumerate(planes) :
            # while it is not converged .....................
            converged = False
            while converged == False :
                old_plane_score = planes_scores[plane_index]
                for atom_in_plane in atoms_in_planes[plane_index] :
                    for atom_index , is_bounded_to_atom_in_plane in enumerate( surface_bounded[atom_in_plane] ):
                        if is_bounded_to_atom_in_plane and ( atom_index not in atoms_in_planes[plane_index] ) :
                            #print( plane_index , ':',  atoms_in_planes[plane_index] + [atom_index] )
                            newplane = best_plane( planes[plane_index] , positions[atoms_in_planes[plane_index] + [atom_index]] , summed_sqd_errors )
                            newplane_score = plane_score_function( newplane , positions[ atoms_in_planes[plane_index] + [atom_index] ] )
                            #print( planes_scores[plane_index] , plane_score_function( newplane , positions[ atoms_in_planes[plane_index] + [atom_index] ] ) )
                            if planes_scores[plane_index] > newplane_score :
                                planes[plane_index] = newplane
                                planes_scores[plane_index] = newplane_score
                                atoms_in_planes[plane_index].append( atom_index )
                if old_plane_score == planes_scores[plane_index] :
                    converged = True
                    atoms_in_planes[plane_index] = np.sort(atoms_in_planes[plane_index]).tolist()

        # Planes clusterizations:
        planes_cluster = DBSCAN(eps=0.07 , min_samples=1 ).fit(np.array(planes))
        cluster_indexes_array , cluster_size = np.unique( planes_cluster.labels_ , return_counts=True)
        print( '    Quantity of initial planes clusters:' , np.max( cluster_indexes_array ) + 1 )

        # Planes with equal atoms:
        eq_planes = []
        eq_atoms_in_planes = []
        eq_planes_scores = []
        for plane_index in range( 0 , len(planes) ) :
            should_be_added = True
            for eq_plane_index , eq_plane in enumerate( eq_planes ) :
                if atoms_in_planes[plane_index] == eq_atoms_in_planes[eq_plane_index] : should_be_added = False
            if should_be_added :
                eq_planes.append( planes[plane_index] )
                eq_atoms_in_planes.append( atoms_in_planes[plane_index] )
                eq_planes_scores.append( planes_scores[plane_index] )
        print('    Quantity of planes with different atoms:' , len(eq_planes) )

        # analysing atoms in planes
        print('    analysing atoms in planes...')
        planes_in_each_atom = []
        qtn_planes_in_each_atom = np.zeros(len(positions) , dtype=int)
        for atom_index in range(0 , len(positions)) :
            planes_in_each_atom.append( [ ] )
            for eq_plane_index in range(0 , len(eq_planes) ):
                if atom_index in eq_atoms_in_planes[eq_plane_index] :
                    planes_in_each_atom[atom_index].append( eq_plane_index  )
                    qtn_planes_in_each_atom[atom_index] += 1
        print('    bag_qtn_planes_in_each_atom: ' + '[' + ','.join(np.array(qtn_planes_in_each_atom, dtype=str)) + ']' )
        print('    reg_qtn_atoms_in_arestas:' , np.sum(qtn_planes_in_each_atom == 2 ) )
        print('    reg_qtn_atoms_in_vertices:' , np.sum(qtn_planes_in_each_atom > 2 ) )
        print('    reg_qtn_atoms_in_planes:' , np.sum(qtn_planes_in_each_atom == 1 ) )
        qtn_atoms_shered_by_planes = np.zeros([len(eq_planes), len(eq_planes)] , dtype=int)
        for plane_index_1 in range(0 , len(eq_planes) ):
            for plane_index_2 in range(0 , len(eq_planes) ):
                if plane_index_1 > plane_index_2 :
                    for atom_plane_1_index in eq_atoms_in_planes[plane_index_1] :
                        for atom_plane_2_index in eq_atoms_in_planes[plane_index_2] :
                            if atom_plane_1_index == atom_plane_2_index :
                                qtn_atoms_shered_by_planes[plane_index_1,plane_index_2] += 1
                                qtn_atoms_shered_by_planes[plane_index_2,plane_index_1] += 1


        # analysing planes
        print('    analysing planes properties...')
        plane_angs = np.zeros([len(eq_planes), len(eq_planes)])
        for plane_index_1 in range(0 , len(eq_planes) ):
            for plane_index_2 in range(0 , len(eq_planes) ):
                if plane_index_1 > plane_index_2:
                    ang = two_planes_angle(eq_planes[plane_index_1],eq_planes[plane_index_2])
                    plane_angs[plane_index_1, plane_index_2] = ang*1
                    plane_angs[plane_index_2, plane_index_1] = ang*1
        bag_vertices_angles = np.zeros( len(positions) )
        bag_arestas_angles = np.zeros( len(positions) )
        for atom_index , planes_list in enumerate(planes_in_each_atom):
            if qtn_planes_in_each_atom[atom_index] < 2 : # atom on core ( == 0) or atom on plane surfaca ( == 1)
                continue
            if qtn_planes_in_each_atom[atom_index] == 2 : # atom on aresta
                bag_arestas_angles[atom_index] = np.pi - plane_angs[planes_list[0],planes_list[1]]
            if qtn_planes_in_each_atom[atom_index] > 2 : # atom on vertice
                line_versor = np.zeros(3)
                for plane_index_1 in planes_list :
                    line_versor += eq_planes[plane_index_1][0:3] / np.linalg.norm( eq_planes[plane_index_1][0:3] )
                line_versor = line_versor / np.linalg.norm(line_versor)
                angle_line_planes = np.zeros(qtn_planes_in_each_atom[atom_index])
                for aux_index, plane_index_1 in enumerate(planes_list) :
                    angle_line_planes[aux_index] = np.arcsin( np.sum(line_versor * eq_planes[plane_index_1][0:3]) / (np.linalg.norm(eq_planes[plane_index_1][0:3])*np.linalg.norm(line_versor)) )
                bag_vertices_angles[atom_index] = 2 * np.average( angle_line_planes )
        print('    bag_arestas_angles: ' + '[' + ','.join(np.array( bag_arestas_angles, dtype=str)) + ']' )
        print('    bag_vertices_angles: ' + '[' + ','.join(np.array( bag_vertices_angles, dtype=str)) + ']' )

    else :
        print('    bag_qtn_planes_in_each_atom: ' + '[' + ','.join(np.array(np.zeros(Qtna), dtype=str)) + ']' )

def summed_abs_errors(params, xyz):
    """
    It meansure the errors distance betweem a plane and points, with a L1 norm.

    Parameters
    ----------
    params: numpy array of floats, with lengh 4.
            params of index 0, 1, and 2 multiply variables x, y, and z
            respectively in the plane equation:
            x*params[0] + y*params[1] + z*params[2] + params[3] = 0.

    xyz: numpy array of floats (n,3) shaped.
         Cartezian positions of the points.

    Return
    ------
    l1_norm: floats.
             The L1 norm of the distance between point and plane.
    """

    dist = params[3]
    abc = params[0:3]
    aux = np.sqrt(np.sum(abc**2))
    return np.sum(np.abs(np.sum(xyz*abc, axis=1) + dist)/aux)


def summed_sqd_errors(params, xyz):
    """
    It meansure the errors distance betweem a plane and points, with a L2 norm.

    Parameters
    ----------
    params: numpy array of floats, with lengh 4.
            params of index 0, 1, and 2 multiply variables x, y, and z
            respectively in the plane equation:
            x*params[0] + y*params[1] + z*params[2] + params[3] = 0.

    xyz: numpy array of floats (n,3) shaped.
         Cartezian positions of the points.

    Return
    ------
    l2_norm: floats.
             The L2 norm of the distance between point and plane.
    """

    dist = params[3]
    abc = params[0:3]
    aux = np.sqrt(np.sum(abc**2))
    return np.sum(((np.sum(xyz*abc, axis=1) + dist) / aux)**2)


def best_plane(params, xyz, error_function):
    """
    This function find the best Ax + By + Cd  + D = 0 equation, were x,y,z
    are points in the plane, and A, B, C, D are the plane equation parameters.
    For basic ideas, see:  https://mathinsight.org/distance_point_plane.
    Parameters
    ----------
    params: numpy array of floats, with lengh 4.
            trail parameters, A, B, C, D are the params[i] for i= 0, 1, and 2,
            respectively in the plane equation:
            x*params[0] + y*params[1] + z*params[2] + params[3] = 0.
    xyz: numpy array of floats (n,3) shaped.
         Cartezian positions of the points.
    error_function: function.
                    A function that meansure the plane-points distance.

    Return
    ------
    best_plane_params: numpy array of floats, with lengh 4.
                       similar to params, but for the plane that best fit the
                       points.
    """

    results = optimize.minimize(error_function, params, args=(xyz),
                                method="L-BFGS-B", tol=1.E-7,
                                options={"maxiter": 50, "disp": False})
    if not results.success:
        print("WARNING: THE PLANE FITTING DO NOT SUCCESSED")
    plane = results.x

    # normalizing d to:
    # plane * np.sign(plane[3]) / np.linalg.norm( plane[0:3] )
    # normalizing abc vector to 1 and d to a positive value
    return plane / plane[3]


def plane_score_function(plane, atoms_positions):
    """It calculate a score for a plane with contaion several atons"""
    dist = plane[3]
    abc = plane[0:3]
    aux = np.sqrt(np.sum(abc**2))
    errors = np.abs(np.sum(atoms_positions*abc, axis=1) + dist) / aux
    return np.sum(errors)*1.72 - len(atoms_positions)


def plane_atom_distance(plane, atom_position):
    dist = plane[3]
    abc = plane[0:3]
    aux = np.sqrt(np.sum(abc**2))
    return np.abs(np.sum(atom_position * abc) + dist) / aux


def two_planes_angle(plane_1, plane_2):
    """The angle between two planes.
    See more in https://byjus.com/maths/angle-between-two-planes/

    Parameters
    ----------
    plane_1, plane_2: np.array of floats of lengh 4.
                      Each variable contain the paramenter of a plane like:
                      x*params[0] + y*params[1] + z*params[2] + params[3] = 0.
    Return
    ------
    angle: float.
           The angle between the planes.
    """

    n2n1 = np.sum(plane_1[0:3]*plane_2[0:3])
    norm_n1 = np.linalg.norm(plane_1[0:3])
    norm_n2 = np.linalg.norm(plane_2[0:3])
    return np.arccos(np.abs(n2n1/(norm_n1*norm_n2)))


ATOMS_RADII = {'Ce': 0.97, 'Zr': 0.78, 'O': 1.38, 'Ag': 1.44, 'Au': 1.44,
               'Cd': 1.52, 'Co': 1.25, 'Cr': 1.29, 'Cu': 1.28, 'Fe': 1.26,
               'Hg': 1.55, 'In': 1.67, 'Ir': 1.36, 'Sr': 2.15, 'Th': 1.80,
               'Ti': 1.47, 'Pt': 1.39, 'Rh': 1.34, 'Ru': 1.34, 'Sc': 1.64,
               'V': 1.35, 'W': 1.41, 'Zn': 1.37, 'Os': 1.35, 'Pb': 1.75,
               'Pd': 1.37, 'Ni': 1.35}


def volume(not_finished_yet) :
        printt(pl , 1 , "Calculating volume...")
        max_min=np.zeros([3,2])
        for i in range(0,3):
            max_min[i,0] = max( centered_dots_surface[:,i])
            max_min[i,1] = min( centered_dots_surface[:,i])
        grid_size = np.array( ( max_min[:,0]-max_min[:,1] ) / 0.2 , dtype=int )
        #X,Y,Z=  np.mgrid[ max_min[0,0]:max_min[0,1]:complex(0,grid_size[0]) , max_min[1,0]:max_min[1,1]:complex(0,grid_size[1]) , max_min[2,0]:max_min[2,1]:complex(0,grid_size[2])  ]
        x = np.linspace(max_min[0,0],max_min[0,1], grid_size[0])
        y = np.linspace(max_min[1,0],max_min[1,1], grid_size[1])
        z = np.linspace(max_min[2,0],max_min[2,1], grid_size[2])
        inside_surface = np.zeros( grid_size , dtype=bool )
        for x_i in range(0, len(x)) :
            if x_i in np.linspace( 0 , len(x) , 11 , dtype=int ):
                printt( pl , 1 , '    ' + str( int(round(100 * x_i/len(x)))) + " %"  )
            for y_i in range( 0 , len(y)) :
                for z_i in range( 0 , len(z)) :
                    distfromzero = np.linalg.norm( np.array([x[x_i],y[y_i],z[z_i]]) )
                    if distfromzero < dots_surface_description.minmax[0] : inside_surface[x_i,y_i,z_i]=True
                    if distfromzero > dots_surface_description.minmax[1] : inside_surface[x_i,y_i,z_i]=False
                    if distfromzero < dots_surface_description.minmax[1] and distfromzero > dots_surface_description.minmax[0] :
                        distances = cdist(np.array([[x[x_i],y[y_i],z[z_i]]]) , positions ).flatten()
                        if any( distances - atoms_path_dot_touch_distance < 0 ) :
                            inside_surface[x_i,y_i,z_i]=True
        printt( pl , 1 , "    Cluster volume: " + str( np.prod(max_min[:,0]-max_min[:,1]) * np.sum(inside_surface) / np.prod(np.shape(inside_surface)) ) )


def ellipsoid_fit(position):
    """Fitting ellipsoid function from the positions, that shoud be
    distributed around something similar to a ellipsoid.
    I get it from: https://github.com/marksemple/pyEllipsoid_Fit
    See also: http://www.mathworks.com/matlabcentral/fileexchange/24693-ellipsoid-fit
    But it have a MIT License (MIT)."""
    x = position[:, 0]
    y = position[:, 1]
    z = position[:, 2]
    D = np.array([x * x + y * y - 2 * z * z,
                 x * x + z * z - 2 * y * y,
                 2 * x * y,
                 2 * x * z,
                 2 * y * z,
                 2 * x,
                 2 * y,
                 2 * z,
                 1 - 0 * x])
    d2 = np.array(x * x + y * y + z * z).T # rhs for LLSQ
    u = np.linalg.solve(D.dot(D.T), D.dot(d2))
    a = np.array([u[0] + 1 * u[1] - 1])
    b = np.array([u[0] - 2 * u[1] - 1])
    c = np.array([u[1] - 2 * u[0] - 1])
    v = np.concatenate([a, b, c, u[2:]], axis=0).flatten()
    A = np.array([[v[0], v[3], v[4], v[6]],
                  [v[3], v[1], v[5], v[7]],
                  [v[4], v[5], v[2], v[8]],
                  [v[6], v[7], v[8], v[9]]])

    center = np.linalg.solve(- A[:3, :3], v[6:9])

    translation_matrix = np.eye(4)
    translation_matrix[3, :3] = center.T

    R = translation_matrix.dot(A).dot(translation_matrix.T)

    evals, evecs = np.linalg.eig(R[:3, :3] / -R[3, 3])
    evecs = evecs.T

    radii = np.sqrt(1. / np.abs(evals))
    radii *= np.sign(evals)

    return center, evecs, radii


def compute_bounds_dimp(positions, chemical_symbols, print_convergence=True):
    """This function compute the effective coordination number and the average
    bound distance.

    Parameters
    ----------
    positions: numpy.array, (3,n) shaped.
               The cartezian positions of the atoms.
    chemical_symbols : numpy.array, (n,) shaped.
                       the chemical symbols of each atom.
    criteria: float greater than zero.
              Creteria for connectivity.
    print_convergence : Float, optional.

    Returns
    -------
    ecn, dav: numpy.array, (n,) shaped.
              They contain the calculated ecn and dav for each atom.
    Pij: numpy.array, (n,n) shaped.
         The index of connectivity between pairs of atoms.
    """

    dij = cdist(positions, positions) + 100*np.eye(len(positions))

    dij_min = np.array([[ bonds_distance( a1 , a2 )[0] for a1 in chemical_symbols] for a2 in chemical_symbols ])
    dij_max = np.array([[ bonds_distance( a1 , a2 )[1] for a1 in chemical_symbols] for a2 in chemical_symbols ])
    Pij = np.logical_and( ( dij < dij_max ) , ( dij > dij_min) )
    ecn = np.sum(Pij, axis=1)
    ecn_zeros_to_one = np.array( ecn == 0 , dtype=int )
    dav = np.sum(Pij * dij , axis=1) / (ecn + ecn_zeros_to_one )
    ri= dav/2.
    def cost( ri , dij , Pij ):
        ri_sum_rj = ri.reshape([1,-1]) + ri.reshape([-1,1])
        return np.sum(((dij - ri_sum_rj)*Pij)**2)
    results = optimize.minimize( cost , ri , args=( dij, Pij ) , method="L-BFGS-B" , tol=1.E-7 , options={"maxiter":50 , "disp":False})
    rav = results.x
    if print_convergence : print( '    bag_ecn: ' + '['+','.join(np.array( ecn , dtype=str))+']' )
    if print_convergence : print( '    bag_dav: ' + '['+','.join(np.array( rav , dtype=str))+']' )
    if print_convergence : print( '    ECN analysis finished...' )
    return ecn , rav , Pij


def compute_bounds ( positions , chemical_symbols , criteria, print_convergence = True ):
    """This function compute the effective coordination number and the average
    bound distance.

    Parameters
    ----------
    positions: numpy.array, (3,n) shaped.
               The cartezian positions of the atoms.
    chemical_symbols : numpy.array, (n,) shaped.
                       the chemical symbols of each atom.
    criteria: float greater than zero.
              Creteria for connectivity.
    print_convergence : Float, optional.

    Returns
    -------
    ecn, dav: numpy.array, (n,) shaped.
              They contain the calculated ecn and dav for each atom.
    Pij: numpy.array, (n,n) shaped.
         The index of connectivity between pairs of atoms.
    """

    if print_convergence : print("\n\nBounds analysis:")
    dij = cdist(positions,positions) + 100*np.eye(len(positions))
    chem_to_radius_dict = { 'H' : 0.53 , 'C' : 0.67 , 'Fe' : 1.2427215683486652 , 'Co' : 1.1692352848082848 , 'Ni' : 1.183294988681055 , 'Cu' : 1.2287289300466713}
    rexp = np.array([ chem_to_radius_dict[i] for i in chemical_symbols ])
    ri_rj_sum = rexp.reshape([1,-1]) + rexp.reshape([-1,1])
    Pij = abs(dij - ri_rj_sum)/(ri_rj_sum) < criteria
    ecn = np.sum(Pij, axis=1)
    dav = np.sum(Pij * dij , axis=1) / np.sum(Pij,axis=1)
    if print_convergence : print( '    bag_ecn: ' + '['+','.join(np.array( ecn , dtype=str))+']' )
    if print_convergence : print( '    bag_dav: ' + '['+','.join(np.array( dav , dtype=str))+']' )
    if print_convergence : print( '    Bound analysis finished...' )
    return ecn , dav , np.array(Pij ,dtype=int)


#ADDD to the other
## Fitting ellipsoid
if False : # it work but not always
    extended_atoms_positions = []
    for atom_index in range( 0 , Qtna ) :
        extended_atoms_positions.append( ( np.linalg.norm(positions[atom_index]) + atoms_radii[atom_index] ) * positions[atom_index] / np.linalg.norm(positions[atom_index]) )
    extended_atoms_positions=np.array(extended_atoms_positions)
    surface_atoms_positions = extended_atoms_positions[is_surface]
    printt( pl , 1 , '\nFitting a ellipsoid... ')
    center, evecs, radii = ellipsoid_fit( surface_atoms_positions )
    printt( pl , 1 , '  center: ' + str(center) )
    printt( pl , 1 , '  evec 1: ' + str(evecs[0]) )
    printt( pl , 1 , '  evec 2: ' + str(evecs[1]) )
    printt( pl , 1 , '  evec 3: ' + str(evecs[2]) )
    printt( pl , 1 , '  radii: ' + str(radii) )


    # finding elipsoid surface: I implemented the equation 41 of :
    ### NEW ZEALAND JOURNAL OF MATHEMATICS / Volume 34 (2005), 165-198 / SURFACE AREA AND CAPACITY OF ELLIPSOIDS IN n DIMENSIONS / Garry J. Tee
    ### for legendre polynomial I am usining scipy.special.legendre, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.legendre.html
    ### and also http://mathworld.wolfram.com/LegendrePolynomial.html
    printt( pl , 1 , '\nCalculating surface area of the ellipsoid' )
    c , b , a = np.sort(radii) #  are the semiaxis lengh
    alpha=np.sqrt(1-(b/a)**2)
    betha=np.sqrt(1-(c/a)**2)
    legendre_x_value=(alpha**2 + betha**2)/(2*alpha*betha)
    sumation = 0.
    presumation = 0.
    v=0
    printt( pl , 2 , "%16s   %16s   %16s   %10s   %8s" % ( "legendre_degree" , "legendre_factor" , "cocient_factor" , "sumation" , "Delta" ) )
    while abs(sumation - presumation) > 10**-5 or v < 5 :
        if v > 0:
            presumation = sumation * 1.
        legendre_factor = np.sum((legendre_x_value**np.arange(v+1))[::-1] * legendre(v).c )
        cocient_factor = ((alpha*betha)**v)/(1-4*v**2)
        sumation +=  4*np.pi*a*b * cocient_factor * legendre_factor
        printt( pl , 2 , "%16d   %16.2f   %3.12f   %5.10f   %2.5f" % ( v , legendre_factor , cocient_factor , sumation , sumation - presumation ) )
        v+=1
    printt( pl , 1 , "Surface area (ellipsoid): " + str(sumation) )

    ##### Printin Labeled atoms + ellipsoid surface
    printt( pl , 1 , "\nWritting the ellipsoid surface: labeled_structure_plus_dots.xyz")
    spherical_dots = RegRDS_set(1, 900)
    dummy_atoms_to_ellipsoid_suface = np.dot(evecs*radii , spherical_dots.T ).T
    dummy_formula = ''.join(new_chemical_symbols) + 'H'*len(dummy_atoms_to_ellipsoid_suface)
    dummy_positions = list(map(tuple, np.append( positions , dummy_atoms_to_ellipsoid_suface , axis=0 )))
    labeled_structure_plus_ellipsoid = ase.Atoms( dummy_formula , dummy_positions )
    ase.io.write( output_prefix+'labeled_structure_plus_ellipsoid.xyz' , labeled_structure_plus_ellipsoid )


def surf_nearest_atom_atom(positions, sampling_distance,
                           write_shell_points=True) :
    """Algorithm 1 for classify atoms if atoms are in the surface.
    A shell of points is created by projecting the positions of each atom to a
    radius of sampling_distance.
    The suface atoms are the nearest atom to each point in the shell."""
    positions = positions - np.average(positions, axis=0)
    dots = []
    for atom_index_a in range(0 , len(positions)):
        dots.append( sampling_distance * positions[atom_index_a] / np.linalg.norm(positions[atom_index_a]) )
    dots = np.array(dots)
    if write_shell_points : writing_points_xyz('dots_alg1_founded.xyz' , dots)
    printt( pl , 1 , '    trajectories: ' + str( len(dots) )  )
    for counter , dot in zip( range(0, len(dots)) , dots ) :
        distances_from_dot = cdist( np.array([dot]) , positions )
        surface_atom_index = np.argmin( distances_from_dot )
        incidence[ surface_atom_index ] += 1
        if is_surface[ surface_atom_index ] == False :
            is_surface[ surface_atom_index ] = True
            printt( pl , 2 , '    adding: ' + str( surface_atom_index) )
        if counter in np.linspace(0, len(dots) , 11 ,dtype=int )[0:-1]  :
            printt( pl , 1 , '        ' + str(round(100*counter/len(dots))) + '%' )

def surf_nearest_random(positions, sampling_distance, max_cicle,
                        write_shell_points=True) :
    """Algorithm 2 for classify atoms if atoms are in the surface.
    A shell of points is created by random over a surface of a radius equal to
    sampling_distance.
    The suface atoms are the nearest atom to each point in the shell."""
    positions = positions - np.average(positions, axis=0)
    dots = RandRDS_set( sampling_distance , max_cicle )
    write_points_xyz(output_prefix+'dots_alg2_founded.xyz' , dots )
    printt( pl , 1 , '    trajectories: ' + str( len(dots) )  )
    for counter , dot in enumerate( dots ) :
        distances_from_dot = cdist( np.array([dot]) , positions )
        surface_atom_index = np.argmin( distances_from_dot )
        incidence[ surface_atom_index ] += 1
        if is_surface[ surface_atom_index ] == False :
            is_surface[ surface_atom_index ] = True
            printt( pl , 2 , '    adding: ' + str( surface_atom_index ) )
        if counter in np.linspace(0, len(dots) , 11 ,dtype=int )[0:-1]  :
            printt( pl , 1 , '        ' + str(round(100*counter/len(dots))) + '%' )

def surf_path_random_atom(positions, sampling_distance, max_cicle,
                          write_shell_points=True) :
    """Algorithm 3 for classify atoms if atoms are in the surface.
    A shell of points is created by random over a surface of a radius equal to
    sampling_distance.
    The suface atoms are firt atom found in the paths between each point in the
    shell and each atom."""
    positions = positions - np.average(positions, axis=0)
    dots = []
    Qtna = max_cicle
    for atom_index_a in range(0 , len(positions)):
        dots.append( sampling_distance * positions[atom_index_a] / np.linalg.norm(positions[atom_index_a]) )
    dots = np.array(dots)
    write_points_xyz('output_prefix+dots_alg3_founded.xyz' , dots )
    printt( pl , 1 , '    trajectories: ' + str( len(dots)*Qtna ) )
    for counter, dot in enumerate(dots) :
        for atom_b_index in range(0, Qtna) :
            Qtnsteps = int(round( np.linalg.norm( dot - positions[atom_b_index] ) / 0.1 ))
            path = linspace_r3_vector(dot, positions[atom_b_index], Qtnsteps)
            path_break = False
            for dot in path :
                for atom_index in range(0 , Qtna) :
                    if dot_in_atom(dot, positions[atom_index], atoms_path_dot_touch_distance[atom_index]) :
                        incidence[atom_index] += 1
                        if is_surface[atom_index] == False :
                            is_surface[atom_index] = True
                            printt(pl, 2, '    adding: ' + str(atom_index))
                        path_break = True
                        break
                if path_break : break
        if counter in np.linspace(0, len(dots), 11, dtype=int)[0:-1] :
            printt(pl, 1, '        ' + str(round(100*counter/len(dots))) + '%')


def surf_path_2atoms_atom(positions, sampling_distance, write_shell_points=True) :
    """Algorithm 4 for classify atoms if atoms are in the surface.
    A shell of points is created in direction of the sum each two atoms
    positions with a radius equal to sampling_distance.
    The surface atoms are colection of the firt atom found in the paths between
    each point in the shell and each atom."""
    positions = positions - np.average(positions, axis=0)
    dots = []
    for atom_index_a in range(0 , len(positions)):
        for atom_index_b in range( atom_index_a , len(positions)):
            position_sum = positions[atom_index_a] + positions[atom_index_b]
            dots.append( sampling_distance * position_sum / np.linalg.norm(position_sum) )
    dots = np.array(dots)
    wriite_points_xyz(output_prefix+'dots_alg4_founded.xyz' , dots )
    printt(pl, 1, '    trajectories: ' + str(len(dots)*Qtna))
    for counter, dot in enumerate(dots) :
        for atom_b_index in range(0, Qtna ):
            Qtnsteps = int(round( np.linalg.norm( dot - positions[atom_b_index] ) / step_size ))
            path = linspace_r3_vector( dot , positions[atom_b_index] , Qtnsteps )
            path_break = False
            for path_dot in path :
                for atom_index in range( 0 , Qtna ) :
                    if dot_in_atom( path_dot , positions[atom_index] , atoms_path_dot_touch_distance[atom_index] ) :
                        incidence[ atom_index ] += 1
                        if is_surface[ atom_index ] == False :
                            is_surface[ atom_index ] = True
                            printt( pl , 2 , '    adding: ' + str(atom_index ) )
                        path_break = True
                        break
                if path_break : break
        if counter in np.linspace(0, len(dots) , 11 ,dtype=int )[0:-1]  :
            printt( pl , 1 , '        ' + str(round(100*counter/len(dots))) + '%' )


def surf_path_random_atom(positions, sampling_distance, write_shell_points=True) :
    """Algorithm 5 for classify atoms if atoms are in the surface.
    A shell of random points of radius equal to sampling_distance is created.
    The surface atoms are a colection of the firt atom found in the paths
    between each point in the shell and each atom."""
    positions = positions - np.average(positions, axis=0)
    alg_5_max_cicle = max_cicle * 1
    dots = RandRDS_set( sampling_distance , alg_5_max_cicle )
    write_points_xyz(output_prefix+'dots_alg5_founded.xyz' , dots )
    printt( pl , 1 , '    trajectories: ' + str( len(dots) * Qtna) )
    for counter , dot in enumerate(dots) :
        for atom_b_index in range(0, Qtna ):
            Qtnsteps = int(round( np.linalg.norm( dot - positions[atom_b_index] ) / step_size ))
            path = linspace_r3_vector( dot , positions[atom_b_index] , Qtnsteps )
            path_break = False
            for path_dot in path :
                for atom_index in range( 0 , Qtna ) :
                    if dot_in_atom( path_dot , positions[atom_index] , atoms_path_dot_touch_distance[atom_index] ) :
                        incidence[ atom_index ] += 1
                        if is_surface[ atom_index ] == False :
                            is_surface[ atom_index ] = True
                            printt( pl , 2 , '    adding: ' + str(atom_index ) )
                        path_break = True
                        break
                if path_break : break
        if counter in np.linspace(0, len(dots) , 11 ,dtype=int )[0:-1]  :
            printt( pl , 1 , '        ' + str(round(100*counter/len(dots))) + '%' )

# Johnatan (june of 2019): The algorithm 6 were removed due it is equal to 5...
