""" An algorithm to analyse the structure and geometry of cluster of atoms."""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.stats import describe
from scipy.spatial.distance import cdist
from scipy.signal import find_peaks

from quandarium.aux import RegRDS_set
from quandarium.aux import large_surfaces_index
from quandarium.aux import write_points_xyz
from quandarium.aux import comp_aveabs
from quandarium.aux import comp_roptl2
from quandarium.aux import comp_gaussian
from quandarium.aux import comp_pij_maxdamped
from quandarium.aux import comp_minmaxbond
from quandarium.aux import comp_rs
from quandarium.aux import logistic


def avradius(positions, raddii=None, useradius=False):
    """Return the average radius for the molecule.
    
    Parameters
    ----------
    positions: numpy array of floats (n,3) shaped.
               Cartezian positions of the atoms, in angstroms.
    atomic_radii: numpy array of floats (optional, default=None).
                  Radius of the atoms, in the same order which they appear in
                  positions, in angstroms.
    useradius: bool, (optional, default=False)
               If true the atomic raddii will be employed to calculate the 
               molecular radius

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


def ecndav_rsopt(positions, cheme, kinfo, Rinfo, roundpijtoecn=False,
                 rcutp=1.15, w=''):
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
    rcutp: float bigger than one.
           The rcut is the radius ori scaled by this parameter.
    roundpijtoecn: If true the pij is rounded to calculate the ecn.

    Returns
    -------
    ecn, dav, ori: numpy.array, (n,) shaped.
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
        R = np.array([Rinfo[e1] for e1 in cheme])
    else:
        R = Rinfo

    if len(positions) == 1:
        ecn = np.array([0.])
        dav = np.array([0.])
        ori = np.array([R[0]])
        pij = np.array([[0.]])
        return ecn, dav, ori, pij

    ori = R * 1.
    rcut = ori * rcutp
    bounds = []

    for ind, _ in enumerate(cheme):
        bounds.append((R[ind]*0.6, R[ind]*1.4))
    bounds = tuple(map(tuple, bounds))

    if w:
        rs_opt = optimize.minimize(comp_rs, ori, args=(dij, k, R, rcutp, w),
                                   bounds=bounds,
                                   method="L-BFGS-B", tol=1.E-6,
                                   options={"maxiter": 250, "disp": False})
    else:
        rs_opt = optimize.minimize(comp_rs, ori, args=(dij, k, R, rcutp),
                                   bounds=bounds,
                                   method="L-BFGS-B", tol=1.E-6,
                                   options={"maxiter": 250, "disp": False})

    if not rs_opt.success:
        print('Otimization has not converged.')

    ori = rs_opt.x
    rcut = ori * rcutp
    pij = logistic(dij, rcut, k)
    ecn = np.sum(pij, axis=1)
    # calcuating dav avoiding division by 0
    ecn_aux = ecn < 1.
    ecn_aux2 = ecn * 1.
    ecn_aux2[ecn_aux] = 1.
    dav = np.sum(dij*pij, axis=1) / ecn_aux2
    if roundpijtoecn:
        ecn = np.sum(np.round(pij), axis=1)

    bounds = np.array(list(map(list, bounds)))[:, 0]
    if np.any(ori == bounds):
        print(ecn[ori == bounds])
        print(ori[ori == bounds])

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
        print("Initializing ECN-Ropt analysis!")

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

    dij_max = np.array([[comp_minmaxbond(a1, a2)[1] for a1 in chemical_symbols]
                        for a2 in chemical_symbols], dtype=float)
    pij_max = dij < dij_max
    step = 0
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
        if print_convergence:
            print('   ', parameter1, parameter2)
        step += 1
    if not results.success:
        print('    Final r optimiation failed! see the massange: '
              '{:s}'.format(results.message))

    if roundpijtoecn:
        ecn = np.sum(np.round(pij), axis=1)

    ori_sum_orj = ori.reshape([1, -1]) + ori.reshape([-1, 1])
    dav = np.sum((ori_sum_orj)*pij, axis=1) / np.sum(pij, axis=1)

    if plot_name:
        bins = 400
        sig = 0.05
        xvalues = np.linspace(0.5, 2, bins)
        rd_per_atom = np.zeros([len(chemical_symbols), bins])
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
            plt.savefig(plot_name + '_' + str(i) + '.png')

    if print_convergence:
        print('    Analysis concluded!')

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
    return is_surface, exposition
