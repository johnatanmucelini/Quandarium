# An algorithm to analyse the structure and geometry of cluster of atoms

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

parser = argparse.ArgumentParser(description='Analyse the geometry of clusters. Input formats: .xyz, geometry.in. ')
parser.add_argument( '--input'       , dest='input_file'  , action='store' , type=str    , help='Defines the input file.' , required=True )
parser.add_argument( '--output_prefix' , dest='output_prefix' , action='store' , default='' , type=str     , help='Defines the output file prefix (default=none).' , required=False )
parser.add_argument( '--print_level' , dest='print_level' , action='store' , default=2   ,type=int    , choices=[0,1,2]   , help='Large arguments allow to print more output information (default=1).' , required=False )
parser.add_argument( '--algorithms'  , dest='algorithms'  , action='store' , default=[0,0,0,0,0,0,1]    , type=list         , help='List with 7 elements, 0/1 to turn on/off (X_i != 0) the algorithms (default=\'0,0,0,0,0,0,1\').' , required=False )
parser.add_argument( '--distance'    , dest='distance'    , action='store' , default=4.  , type=float , help='Defines the distance...' , required=False )
parser.add_argument( '--max_cicles'  , dest='max_cicle'   , action='store' , default=200 , type=int   , help='Defines the maximum number of cicles...' , required=False )
parser.add_argument( '--step_size'   , dest='step_size'   , action='store' , default=0.1 , type=float , help='Defines the step size for balistic seachs...' , required=False )
parser.add_argument( '--adatom_radius' , dest='adatom_radius'   , action='store' , default=1. , type=float , help='Defines adatom radius (default=1)' , required=False )
parser.add_argument( '--atoms_type'  , dest='atoms_type'  , action='store' , default=[] , type=list , help='List with elements to analyse, the others will be ignored (default=all, exp: \'Fe,Ni\').', required=False )
parser.add_argument( '--remove_ps' , dest='remove_ps' , action='store' , default=False , help='Defines if pseudosurfaces should be removed (default=True).', required=False )
parser.add_argument( '--bc_degree' , dest='degree' , action='store' , default=3 , type=int , choices=[ -1, 1, 2, 3] , help='The boond degree to print: 1, 2, 3, and -1 for 1st, 2nd, 3rd and all of then, respectively.' , required=False )
parser.add_argument( '--bc_count' , dest='count' , action='store' , default='lines' , type=str , choices=['lines','cicles','backs'] , help='Allow to count neighborns within cicles and backs or lines (default=lines).' , required=False )
args = parser.parse_args()


input_file = args.input_file
output_prefix = args.output_prefix
pl = args.print_level
algorithms = args.algorithms
distance   = args.distance
max_cicle  = args.max_cicle
step_size  = args.step_size
adatom_radius = args.adatom_radius
atoms_type = ''.join(args.atoms_type)
remove_ps = args.remove_ps
degree = args.degree
count = args.count

np.random.seed(1234)

def compute_bounds_dimp ( positions , chemical_symbols , criteria, print_convergence = True ):
    if print_convergence : print("\n\nBounds analysis:")
    dij = cdist(positions,positions) + 100*np.eye(len(positions))
    def bonds_distance( a1 , a2 ):
        if ((a1 == 'H')  and (a2 == 'H'))  or ((a2 == 'H')  and (a1 == 'H'))  : return [ 0.70 , 1.19 ]
        if ((a1 == 'C')  and (a2 == 'H'))  or ((a2 == 'C')  and (a1 == 'H'))  : return [ 0.90 , 1.35 ]
        if ((a1 == 'C')  and (a2 == 'C'))  or ((a2 == 'C')  and (a1 == 'C'))  : return [ 1.17 , 1.51 ]
        if ((a1 == 'Fe') and (a2 == 'H'))  or ((a2 == 'Fe') and (a1 == 'H'))  : return [ 1.2  , 1.99 ]
        if ((a1 == 'Fe') and (a2 == 'C'))  or ((a2 == 'Fe') and (a1 == 'C'))  : return [ 1.2  , 2.15 ]
        if ((a1 == 'Fe') and (a2 == 'Fe')) or ((a2 == 'Fe') and (a1 == 'Fe')) : return [ 2.17 , 2.8  ]
        if ((a1 == 'Ni') and (a2 == 'H'))  or ((a2 == 'Ni') and (a1 == 'H'))  : return [ 1.2  , 1.98 ]
        if ((a1 == 'Ni') and (a2 == 'C'))  or ((a2 == 'Ni') and (a1 == 'C'))  : return [ 1.2  , 2.08 ]
        if ((a1 == 'Ni') and (a2 == 'Ni')) or ((a2 == 'Ni') and (a1 == 'Ni')) : return [ 2.07 , 2.66 ]
        if ((a1 == 'Co') and (a2 == 'H'))  or ((a2 == 'Co') and (a1 == 'H'))  : return [ 1.2  , 1.91 ]
        if ((a1 == 'Co') and (a2 == 'C'))  or ((a2 == 'Co') and (a1 == 'C'))  : return [ 1.2  , 2.20 ]
        if ((a1 == 'Co') and (a2 == 'Co')) or ((a2 == 'Co') and (a1 == 'Co')) : return [ 2.05 , 2.63 ]
        if ((a1 == 'Cu') and (a2 == 'H'))  or ((a2 == 'Cu') and (a1 == 'H'))  : return [ 1.2  , 1.98 ]
        if ((a1 == 'Cu') and (a2 == 'C'))  or ((a2 == 'Cu') and (a1 == 'C'))  : return [ 1.2  , 2.14 ]
        if ((a1 == 'Cu') and (a2 == 'Cu')) or ((a2 == 'Cu') and (a1 == 'Cu')) : return [ 2.15 , 2.76 ]

        if ((a1 == 'Ce') and (a2 == 'O'))  or ((a2 == 'Ce') and (a1 == 'O'))  : return [ 1.1  , 2.7  ]
        if ((a1 == 'Zr') and (a2 == 'O'))  or ((a2 == 'Zr') and (a1 == 'O'))  : return [ 1.1  , 2.7  ]
        if ((a1 == 'O')  and (a2 == 'O'))  or ((a2 == 'O')  and (a1 == 'O'))  : return [ 1.1  , 2.7  ]
        if ((a1 == 'Ce') and (a2 == 'Ce')) or ((a2 == 'Ce') and (a1 == 'Ce')) : return [ 1.1  , 2.7  ]
        if ((a1 == 'Zr') and (a2 == 'Zr')) or ((a2 == 'Zr') and (a1 == 'Zr')) : return [ 1.1  , 2.7  ]
        if ((a1 == 'Zr') and (a2 == 'Ce')) or ((a2 == 'Zr') and (a1 == 'Ce')) : return [ 1.1  , 2.7  ]

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

def compute_ecn_dav_ropt(positions, chemical_symbols, pij_to_int=True) :
    """Return the effective coordination number (ecn) and the atomic radius ri
    and the conective index matrix Pijself.
    positions: np.array of (n,3) shape with atoms positions.
    chemical_symbols: np.array with the atoms symbols strings.
    pij_to_int: around the conective index matrix to intiger."""

    logging.debug("Initializing ECN-Ropt analysis!")

    if (type(positions) != np.ndarray)
      or (np.shape(positions)[1] != 3)
      or (type(positions[0][0]) != bool) :
        logging.error("positions must be a (n,3) shaped numpy.ndarray!"
                      + " Aborting...")
        sys.exit(1)

    if (type(chemical_symbols) != np.ndarray)
      or (type(chemical_symbols[0][0]) != str) :
        logging.error("chemical_symbols must be a numpy.ndarray of strings.")
        sys.exit(1)

    dij = cdist(positions,positions) + 100*np.eye(len(positions))
    dav = np.max( cdist(positions,positions) , axis=0)
    ri = dav / 2.
    ri_pre = np.zeros_like(ri)
    ecn_pre = np.zeros_like(dij)
    def compute_pij_max_dumbed( dij, ri , Pij_max ):
        return np.exp(1-(dij / ( ri.reshape(-1,1) + ri.reshape(1,-1)   ))**6 -(dij/3.5)**4)*Pij_max

    dij_max = np.array([[ bonds_distance( a1 , a2 )[1] for a1 in chemical_symbols] for a2 in chemical_symbols ])
    Pij_max = dij < dij_max
    v=0
    if print_convergence : print("    \Delta sum_i(abs(r_i))/N    \Delta sum_i(abs(ECN_i))/N")
    while np.sum(np.abs(ri_pre - ri ))/ len(ri) > 10E-8 or v < 2 :
        if v > 0 :
            ri_pre = ri * 1.
            ecn_pre = ecn * 1.
        Pij = compute_pij_max_dumbed( dij, ri , Pij_max )
        def cost_l2 ( ri , dij , Pij ):
            ri_sum_rj = ri.reshape([1,-1]) + ri.reshape([-1,1])
            return np.sum(((dij - ri_sum_rj)*Pij)**2)
        results = optimize.minimize( cost_l2 , ri , args=( dij, Pij ) , bounds=((0.5,1.7),)*len(chemical_symbols) , method="L-BFGS-B" , tol=1.E-7 , options={"maxiter":50 , "disp":False})
        ri = results.x
        #ri_dav = np.sum(Pij * dij , axis=1) / np.sum(Pij,axis=1)
        #ri = (ri + ri_dav) / 2.
        ecn = np.sum(Pij, axis=1)
        if print_convergence : print( '   ' ,  np.sum(np.abs(ri_pre - ri ))/ len(ri) , np.sum(np.abs(ecn - ecn_pre))/len(ecn) )
        #print( ri )
        v+=1
    #if pij_to_int == True :
    #    Pij = np.round(Pij)
    #    results = optimize.minimize( cost_l2 , ri , args=( dij, Pij ) , method="L-BFGS-B" , tol=1.E-7 , options={"maxiter":50 , "disp":False})
    #    ri = results.x
    #    ecn = np.sum(Pij, axis=1)
    if print_convergence : print( '    bag_ecn: ' + '['+','.join(np.array( ecn , dtype=str))+']' )
    if print_convergence : print( '    bag_ri: ' + '['+','.join(np.array( ri , dtype=str))+']' )
    if print_convergence : print( '    ECN analysis finished...' )
    bins=400
    sig=0.05
    x=np.linspace(0.5,2,bins)
    rd_per_atom = np.zeros([len(chemical_symbols),bins])
    ecn_integrated = np.zeros(len(chemical_symbols))
    def gaussian(x, mu, sig):
        return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)
    for i, atomi in enumerate(chemical_symbols):
        for j, atomj in enumerate(chemical_symbols):
                rd_per_atom[i] += gaussian(x,dij[i,j]/(ri[i] + ri[j]),sig)
        peaks, _ = find_peaks( (rd_per_atom[i]+1)**-1, height=0)
        plt.plot(rd_per_atom[i])
        plt.plot(peaks, rd_per_atom[i][peaks], "x")
        plt.plot(np.zeros_like(rd_per_atom[i]), "--", color="gray")
        ecn_integrated[i] = np.trapz( rd_per_atom[i,0:peaks[0]] , x = x[0:peaks[0]] )
    plt.show()
    print(ecn_integrated)
    print(ecn)
    return ecn , ri, Pij

def compute_ecn_dav ( positions , print_convergence=True ):
    if print_convergence : print("\n\nECN analysis:")
    dij = cdist(positions,positions) + 100*np.eye(len(positions))
    dav = np.max( cdist(positions,positions) , axis=0)
    dav_pre = np.zeros_like(dav)
    ecn_pre = np.zeros_like(dij)
    v=0
    if print_convergence : print("    \Delta sum_i(abs(dav_i))/N    \Delta sum_i(abs(ECN_i))/N")
    while np.sum(np.abs(dav_pre - dav ))/ len(dav) > 10E-8 or v < 2 :
        if v > 0 :
            dav_pre = dav * 1.
            ecn_pre = ecn * 1.
        Pij = np.exp(1-(2* dij / ( dav.reshape(-1,1) + dav.reshape(1,-1)   ))**6)
        ecn = np.sum(Pij, axis=1)
        dav = np.sum(Pij * dij,axis=1) / np.sum(Pij,axis=1)
        ecn = np.sum(Pij, axis=1)
        if print_convergence : print( '   ' ,  np.sum(np.abs(dav_pre - dav ))/ len(dav) , np.sum(np.abs(ecn - ecn_pre))/len(dav) )
        v+=1
    if print_convergence : print( '    bag_ecn: ' + '['+','.join(np.array( ecn , dtype=str))+']' )
    if print_convergence : print( '    bag_dav: ' + '['+','.join(np.array( dav , dtype=str))+']' )
    if print_convergence : print( '    ECN analysis finished...' )
    return ecn , dav , Pij

def summed_abs_errors(params, xyz):
    d = params[3]
    abc = params[0:3]
    aux = np.sqrt( np.sum( abc**2 ) )
    return np.sum( np.abs( np.sum(xyz*abc, axis=1) + d ) / aux )

def summed_sqd_errors(params, xyz):
    d = params[3]
    abc = params[0:3]
    aux = np.sqrt( np.sum( abc**2 ) )
    return np.sum( (( np.sum(xyz*abc, axis=1) + d) / aux )**2 )

def best_plane( params, xyz , error_function) :
    # This function find the best Ax + By + Cd  + D = 0  equation, were x,y,z are points in the plane, and A,B,C,D are the plane equation parameters... see https://mathinsight.org/distance_point_plane for basic ideas
    results=optimize.minimize( error_function , params , args=( xyz ) , method="L-BFGS-B" , tol=1.E-7 , options={"maxiter":50 , "disp":False})
    if results.success == False :
        print("\n\nWARNING: THE PLANE FITTING DO NOT SUCCESSED\n\n")
    plane = results.x
    return  plane / plane[3]   # normalizing d to                  plane * np.sign(plane[3]) / np.linalg.norm( plane[0:3] )  # normalizing abc vector to 1 and d to a positive value

def surface_bounded_pij ( pij , is_surface, cutoff ) :
    ecn_bounded_ones = np.array(  pij >= cutoff , dtype=int)
    is_surface_ones = np.array( is_surface, dtype=int)
    return np.array((ecn_bounded_ones * is_surface_ones).T * is_surface_ones , dtype=bool)

def plane_score_function( plane , atoms_positions ) : # calculate a score for a plane with contaion several atons
    d = plane[3]
    abc = plane[0:3]
    aux = np.sqrt( np.sum( abc**2 ) )
    errors = np.abs( np.sum(atoms_positions*abc, axis=1) + d ) / aux
    return np.sum(errors)*1.72 - len(atoms_positions)

def plane_atom_distance( plane , atom_position ) :
    d = plane[3]
    abc = plane[0:3]
    aux = np.sqrt( np.sum( abc**2 ) )
    return np.abs( np.sum( atom_position * abc ) + d ) / aux

def two_planes_angle(plane_1 , plane_2) :
    # see https://byjus.com/maths/angle-between-two-planes/ for more
    return np.arccos( np.abs( np.sum(plane_1[0:3]*plane_2[0:3]) / (np.linalg.norm(plane_1[0:3])*np.linalg.norm(plane_2[0:3])) ) )


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



radius = {
    'Ce' : 0.97 , 'Zr' : 0.78 , 'O' :  1.38 , 'Ag' : 1.44 , 'Au' : 1.44 ,
    'Cd' : 1.52 , 'Co' : 1.25 , 'Cr' : 1.29 , 'Cu' : 1.28 , 'Fe' : 1.26 ,
    'Hg' : 1.55 , 'In' : 1.67 , 'Ir' : 1.36 , 'Sr' : 2.15 , 'Th' : 1.80 ,
    'Ti' : 1.47 , 'Pt' : 1.39 , 'Rh' : 1.34 , 'Ru' : 1.34 , 'Sc' : 1.64 ,
    'V'  : 1.35 , 'W'  : 1.41 , 'Zn' : 1.37 , 'Os' : 1.35 , 'Pb' : 1.75 ,
    'Pd' : 1.37 , 'Ni' : 1.35
    }


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

def surf_path_random_atom(positions, sampling_distance, write_shell_points=True) :
    """Algorithm 7 for classify atoms if atoms are in the surface with surface
    construction.
    Surface is constructed by fullfill the shell of each atoms (atom radius)
    plus a dummy atom radius. (An adatom of this radius could be faoud in each
    point position is the atoms were redig spheres)
    If the point is inside of other atom the point is reject. (The could not be
    inside other atom.)
    The resulting points created one or more surfaces.
    It more than one were found, the system may be composed by more then one
    molecule or there are some empth space whithin the strcture. The dummy
    surfaces cam be removed with the optional variable XXX.
    The surface atoms are the colection of atoms with at least one point in it
    surface."""
    positions = positions - np.average(positions, axis=0)

    dots_try = RegRDS_set( sampling_distance , max_cicle )
    #print_points_xyz(output_prefix+'dots_alg6_founded.xyz' , dots )
    alg_6_max_cicle = len(dots_try) * Qtna
    printt( pl , 1 , '    investigated dots: ' + str(alg_6_max_cicle) )
    dots = positions[0] + RegRDS_set( atoms_path_dot_touch_distance[0] , max_cicle )
    dot_origin = [ [0]* len(dots_try) ]
    for atom_a_index in range( 1 , Qtna ):
        dots = np.append( dots , positions[atom_a_index] + RegRDS_set( atoms_path_dot_touch_distance[atom_a_index] + 0.001 , max_cicle ) , axis=0 )
        dot_origin.append( [atom_a_index]* len(dots_try) )
    dots_atoms_distance = cdist( positions , dots )
    dot_origin = np.array(dot_origin).flatten()
    atomos_radii_projected = np.array([atoms_path_dot_touch_distance]*len(dots)).reshape(len(dots),Qtna).T
    surface_dots = np.sum( dots_atoms_distance < atomos_radii_projected , axis=0 ) == 0
    dots_surface = dots[ surface_dots ]
    # removing small pseudo surfaces
    if remove_ps :
        print( remove_ps )
        dots_for_find_eps = RegRDS_set(max(atoms_path_dot_touch_distance), max_cicle)
        eps = 2.1 * np.max( np.min(cdist( dots_for_find_eps, dots_for_find_eps) + np.eye(len(dots_for_find_eps))*10 , axis=0)  )
        bigger_pseudo_surface_dots = large_surfaces_index(dots_surface, eps)
        dots_surface = dots_surface[ bigger_pseudo_surface_dots ]
        surface_dot_origin = dot_origin[surface_dots][bigger_pseudo_surface_dots]
    else :
        surface_dot_origin = dot_origin[surface_dots]
    #
    surface_atoms , dots_per_surface_atom = np.unique( surface_dot_origin , return_counts=True )
    surface_ratio_per_atom = np.zeros(Qtna)
    incidence = np.zeros(Qtna)
    dots_per_atom = np.zeros(Qtna)
    for atom_index , surface_atom_incidence in zip(surface_atoms ,dots_per_surface_atom) :
         printt( pl , 2 , '    adding: ' + str(atom_index ) )
         is_surface[atom_index] =True
         dots_per_atom[atom_index] = surface_atom_incidence
         incidence[atom_index] = surface_atom_incidence
    write_points_xyz(output_prefix+'dots_alg7_founded.xyz' , dots_surface )
    printt( pl , 1 , "    Surface area (atomic exposition): " + str( sum(atoms_area * incidence / len(dots_try) ) ))
    centered_dots_surface = dots_surface - np.average(dots_surface,axis=0)
    dots_surface_description = describe( cdist( np.array([[0.,0.,0.]]), centered_dots_surface ).flatten() )
    printt( pl , 1 , "    Surface radius (surface dots): " + str( dots_surface_description ) )


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

surface_found

def surface_analysis()...

printt( pl , 1 , '\n\nSurface analysis and statistics: ')
printt( pl , 1 , '    reg_qtn_core_atoms:    ' + str(Qtna - sum(is_surface)) )
printt( pl , 1 , '    reg_qtn_surface_atoms: ' + str(sum(is_surface)) )
surface_radius = []
for atom_index in range(0 , Qtna ):
    if is_surface[atom_index] :
        surface_radius.append( np.linalg.norm(positions[atom_index]) + atoms_radii[atom_index] )
surface_radius = np.array(surface_radius)
printt( pl , 1 , "    Surface atoms radius: " + '['+','.join(np.array( surface_radius , dtype=str))+']' )
printt( pl , 1 , "    Surface atoms radius description: " + str(describe(surface_radius)) )
printt( pl , 1 , "    bag_surface: " + '['+','.join(np.array( is_surface , dtype=str))+']' )
printt( pl , 1 , "    bag_incidence: " + '['+','.join(np.array( incidence , dtype=str))+']' )
printt( pl , 1 , "    bag_incidence_percentage: " + '['+','.join(np.array( incidence * 100 / len(dots_try) , dtype=str))+']' )
printt( pl , 1 , "    bag_surface_area_exposed: " + '['+','.join(np.array( atoms_area * incidence / sum(incidence) , dtype=str))+']' )


def maping_surface_atoms_with_labels ():

## Maping surface labels -- original chemistry elements and writing xyz output files ( work just until 6 different elements)
printt( pl , 1 , '    Writing labeled files: labeled_structure.xyz and labeled_structure_plus_dots.xyz')
original_atoms_labels = np.unique( chemical_symbols )
surface_atom_labels = [ 'He' , 'Ne' , 'Ar' , 'Kr' , 'Xe' , 'Rn' ][0:len(original_atoms_labels)]
for original , new_label in zip( original_atoms_labels , surface_atom_labels ) : print( '        Surface ' + original + ' label: ' + new_label  )
new_chemical_symbols=chemical_symbols.copy()
for atom_index , is_surface_bool in zip( np.arange(0,Qtna) , is_surface ):
    if is_surface_bool == True :
        new_chemical_symbols[atom_index] = surface_atom_labels[ list(original_atoms_labels).index( chemical_symbols[atom_index] ) ]
labeled_structure = ase.Atoms(''.join(new_chemical_symbols) , list(map(tuple, positions)) )
ase.io.write( output_prefix+'labeled_structure.xyz' , labeled_structure )
labeled_structure_plus_dots = ase.Atoms(''.join(new_chemical_symbols) + 'H' +str(len(dots))  , list(map(tuple, np.append( positions , dots ,axis=0 )) ) )
ase.io.write( output_prefix+'labeled_structure_plus_dots.xyz' , labeled_structure_plus_dots )


def printing_fragment_lines():
    atom_index = np.arange(0 ,Qtna )
    print( '    Printing fragment-lines for get_charge.py:' )
    print( '        Surface All: ' + '\'frag' + ','.join(np.array(atom_index[is_surface==True],dtype=str)) + '\'' )
    for chemical_type in np.unique(chemical_symbols) :
        print( '        Surface ' + chemical_type + ": " + '\'frag' + ','.join(np.array(atom_index[(chemical_symbols==chemical_type)*(is_surface==True)],dtype=str)) + '\'' )
    print( '        Core All: ' + '\'frag' + ','.join(np.array(atom_index[is_surface==False],dtype=str)) + '\'' )
    for chemical_type in np.unique(chemical_symbols) :
        print( '        Core ' +chemical_type + ": " + '\'frag' + ','.join(np.array(atom_index[(chemical_symbols==chemical_type)*(is_surface==False)],dtype=str)) + '\'' )


ADDD to the other
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






########################
########################

#### surface topology
printt( pl , 1 , "\n\nSurface atoms topology analysis..." )

# finding surface_bounded_axis
print( 'Bounded criteria: ECN_{ij} < 1.')
surface_bounded = surface_bounded_pij( pij , is_surface , 1. ) # important parameter
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









######################
########################

#### surface topology
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
# end processing
###

###
# begin posprocessing

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
tf_posprocessing = time.time()
print( "\n\n\n\nTotal time:        " + str( tf_posprocessing - ti_header ) )
