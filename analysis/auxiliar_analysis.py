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


def RandRDS_dot( sampling_distance ) :
    """Randon Radial Distributed Samples (dots in R3) Sampling Distance """
    # Randon radial dot sampling / Sphere Point Picking
    # I implemented from http://mathworld.wolfram.com/SpherePointPicking.html
    u , v = np.random.random(2)
    theta = 2 * np.pi * v
    phi = np.arccos(2*u-1)
    x = sampling_distance * np.cos(theta) * np.sin(phi)
    y = sampling_distance * np.sin(theta) * np.sin(phi)
    z = sampling_distance * np.cos(phi)
    Dot = np.array([x,y,z])
    return Dot

def RandRDS_set( sampling_distance , N ) :
    cart_coordinates=[]
    Ncount = 0
    while Ncount < N :
        cart_coordinates.append( RandRDS_dot(sampling_distance) )
        Ncount += 1
    cart_coordinates = np.array(cart_coordinates)
    return cart_coordinates


def RegRDS_set( sampling_distance, N ) :
    # Regular radial dot - set of cartezian points in the surface of a sphere with radius 'sampling_distance'
    # I implemented from https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf
    # How to generate equidistributed points on the surface of a sphere / Markus Deserno
    cart_coordinates=[]
    r = 1
    Ncount = 0
    a = 4. * np.pi * r**2 / N
    d = np.sqrt(a)
    Mtheta = int(round(np.pi/d))
    dtheta = np.pi / Mtheta
    dphi = a / dtheta
    for m in range(0, Mtheta ) :
        theta = np.pi *( m + 0.5 ) / Mtheta
        Mphi = int(round(2 *np.pi * np.sin(theta) / dphi ))
        for n in range( 0 , Mphi ) :
            phi = 2* np.pi * n / Mphi
            Ncount += 1
            y = sampling_distance * np.sin(theta) * np.cos(phi)
            x = sampling_distance * np.sin(theta) * np.sin(phi)
            z = sampling_distance * np.cos(theta)
            cart_coordinates.append([x,y,z])
    cart_coordinates = np.array(cart_coordinates)
    return cart_coordinates

def writing_points_xyz( file_name , positions ) :
    ase.io.write( file_name , ase.Atoms('H'+str(len(positions)) , list(map(tuple,positions)) ))

def linspace_r3_vector ( vector_a , vector_b , Qtnsteps ) :
    return np.array( [ np.linspace( vector_a[0] , vector_b[0] , Qtnsteps ) , np.linspace( vector_a[1] , vector_b[1] , Qtnsteps )  , np.linspace( vector_a[2] , vector_b[2] , Qtnsteps ) ] ).T

def dot_in_atom( dot , atom_position , atom_radius ) :
    if np.linalg.norm( dot - atom_position ) < atom_radius :
        return True
    else:
        return False

def printt( print_level , level, string ):
    if level <= print_level :
        print( string )

def remove_pseudo_surfaces( surface_dots_positions , eps ):
    db = DBSCAN(eps=eps, min_samples=1).fit_predict( surface_dots_positions )
    labels , quanity = np.unique( db , return_counts=True )
    if len(labels) > 1 : print( 'WARNING: ' + str(len(labels)) + ' pseudo_surface(s) was(were) found, sizes: ' + ', '.join(str(quanity).replace('[','').replace(']','').split()) + '\nWARNING: The bigger will be selected!\nWARNING: Please, verify the dots_founded_by_the_algorithm!\n' )
    return db == labels[np.argmax( quanity )]
