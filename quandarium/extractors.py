"""This module present functions that extract data from Quantum Chemistry (QC)
Calculations, and related functions. The QC codes with extractors avaliable
are:
- FHI-aims
- VASP
- formats of ase.io
"""

import re
import pandas as pd
import numpy as np
import ase.io

def extractor_aseengine(outfile):
    """This function open a generical file output (outfile), read it, and
    return some important information.
    Parameters
    ----------
    outfile: string
             The name of the output file.
    Return
    ------
    energy_tot: float.
                The total energy. 

    positions: np.array of floats, (n,3) shaped.
               The cartezian positions of the atoms.

    cheme: np.array of floats, lengh n.
           The chemical species of the structure.

    chemf: str.
           String with the chemical species in alphabetical order
           followed by the quantity of atoms.
    """

    # add tests of the input variables
    molecule = ase.io.read(outfile)

    # Total energy
    energy_tot = molecule.get_total_energy()

    # xyz and
    positions = molecule.positions

    # Chemical Elements
    cheme = molecule.get_chemical_symbols()

    # Chamical formula
    chemf = molecule.get_chemical_formula()

    # NOTE: I will note take other features because it can generate problems
    # if the calculation does not present it...

    return energy_tot, chemf, positions, cheme


def fromfolders_extract_aseengine(folders, output_file_name='aims.out'):
    """It get data from fhi calculation folders.
    Parameters
    ----------
    folders: np.array, list, etc
             The names of the folders with calculations.
    output_file_name: str, (optional, default='aims.out')
                      The output file name.
    Return
    ------
    list_of_new_features_name: list of str 
                               The new data name.
    list_of_new_features_data: list of np.array 
                               The new data.
    """

    # The lists list_of_new_features_data and list_of_new_features_name, colect
    # All the information along the analysis
    list_of_new_features_data = []
    list_of_new_features_name = []

    # getting properties from  extractor_fhi
    etot_list = []
    positions_list = []
    cheme_list = []
    chemf_list = []
    folders = np.array(folders.values)
    print("Initializing data extraction:")
    for index, folder in enumerate(folders):
        etot, chemf, positions, cheme = extractor_aseengine(folder + '/'
                                                            + output_file_name)
        if index % 50 == 0:
            print("    concluded %3.1f%%" % (100*index/len(folders)))
        positions = positions.tolist()
        etot_list.append(etot)
        positions_list.append(positions)
        cheme_list.append(cheme)
        chemf_list.append(chemf)
    print("    concluded %3.1f%%" % (100))
    # Adding new features data and name to its lists
    for new_feature_data in [etot_list, positions_list, cheme_list,
                             chemf_list]:
        list_of_new_features_data.append(new_feature_data)
    for new_feature_name in ['reg_etot', 'bag_positions', 'bag_cheme',
                             'reg_chemf']:
        list_of_new_features_name.append(new_feature_name)

    return list_of_new_features_name, list_of_new_features_data


def extractor_charges(folder, filename='', code='fhi', chargetype='hirs'):
    """This function get the hirshfield charges of an 'fhi' code calculations
    """
    # problema: fileout é o arquivo outcar do vasp e os dados de carga estão
    #           no acf...
    if code == 'VASP':
        atoms = ase.io.read(folder + "POSCAR", format="vasp")
        with open("POSCAR", "r") as file:
            poscar = file.readlines()
        with open("OUTCAR", "r") as file:
            outcar = file.readlines()
        with open("ACF.dat", "r") as file:
            acfdat = file.readlines()

        if chargetype == 'bader':
            badercharge = []
            for i in range(2, len(acfdat) - 4):
                badercharge.append(acfdat[i].split()[4])
            badercharge = np.array(badercharge, dtype=float)

            nuclearcharge = []
            nucleous = []
            for ind, line in enumerate(outcar):
                if "; ZVAL   =" in line:
                    nucleous.append(line.split()[5])
            nucleous = np.array(nucleous, dtype=float)
            quantity_of_each_specie = np.array(poscar[6].split(), dtype=int)
            for ind, line in enumerate(quantity_of_each_specie):
                for j in range(line):  # talvez tenha um problema aqui
                    nuclearcharge.append(nucleous[ind])
            nuclearcharge = np.array(nuclearcharge, dtype=float)

            charges = nuclearcharge - badercharge

    if code == 'fhi':
        if filename == '':
            filename = folder + '/aims.out'
        with open(filename, "r") as file:
            aimsout = file.readlines()

        for line in aimsout:
            if re.findall('.*Number of atoms.*', line):
                number_of_atoms = int(re.search('.*Number of atoms.*',
                                                line).string.split()[5])

        if chargetype == 'mull':
            charges = []
            for ind, line in enumerate(aimsout):
                if "Performing Mulliken charge analysis on all atoms" in line:
                    for jnd in range(ind + 5, ind + 5 + number_of_atoms):
                        charges.append(aimsout[jnd].split()[3])
                    break
            charges = np.array(charges, dtype=float)

        if chargetype == 'hirs':
            charges = []
            aimsout.reverse()
            for ind, line in enumerate(aimsout):
                if "  |   Hirshfeld charge        :" in line:
                    charges.append(line.split()[4])
                    if len(charges) == number_of_atoms:
                        charges.reverse()
                        charges = np.array(charges, dtype=float)
                        break  # it breaks the for...
    return charges


def fromfolders_extract_charges(folders, code, chargetype):
    """This function open VASP and fhi output files, read it, and return the
    atoms charges.
    Parameters
    ----------
    folders: np.array
             A numpy array with the names of the folders with calculations.
    code: str, (one of: 'fhi', 'VASP')
          The code used to the calculation file name.
    chargetype: str, (one of: 'hirs', 'mull','bader')
                The name of the charge analysis. For code='fhi', 'hirs', 'mull'
                are avaliable, while 'bader' is avaliable for code='VASP'.
    Return
    ------
    bag_charges: np.array, of lenght equal to the size of calcfolder
                 It return a bag with the atomic charges
    """

    folders = np.array(folders.values)
    charges_list = []
    print("Initializing fromfolders_extract_charges:")
    for index in range(len(folders)):
        folder = folders[index]
        charges = extractor_charges(folder, code=code, chargetype=chargetype)
        charges_bag = charges.tolist()
        charges_list.append(charges_bag)
        if index % 50 == 0:
            print("    concluded %3.1f%%" % (100*index/len(folders)))
    print("    concluded %3.1f%%" % (100))

    return np.array(charges_list)


def extractor_fhi(outfile='aims.out'):
    """This function open a fhi output file (outfile) and read and return the
    important informations.
    Parameters
    ----------
    outfile: string (optional, default='aims.out')
             The name of the output file.
    Return
    ------
    gap, homo, lumo, mag, energy_tot: floats.
                                      Information of gap, HOMO, LUMO,
                                      magnetization, and total energy,
                                      respectively.
    positions: np.array of floats, (n,3) shaped.
               The cartezian positions of the atoms.
    chemical_species: np.array of floats, lengh n.
                      The chemical species of the structure.
    chemical_formula: str.
                      String with the chemical species in alphabetical order
                      followed by the quantity of atoms.
    folder_name: str.
                 The completely name of the folder.
    """

    # The test of the input variables are necessary.

    with open(outfile, mode='r') as file:
        data = file.readlines()

    # Gap
    for line in reversed(data):
        if re.findall('Overall HOMO-LUMO gap:\s+\d+\.\d+', line):
            gap = float(re.search('Overall HOMO-LUMO gap:\s+\d+\.\d+',
                                  line).string.split()[3])
            break

    # HOMO
    for line in reversed(data):
        if re.findall('^(\s+\d+)(\s+1\.00000)(\s+(|\-)\d+\.\d+){2}', line):
            homo = float(
                re.search('^(\s+\d+)(\s+1\.00000)(\s+(|\-)\d+\.\d+){2}.*',
                          line).string.split()[3])
            break

    # LUMO
    for index, line in enumerate(reversed(data)):
        if re.findall('^(\s+\d+)(\s+1\.00000)(\s+(|\-)\d+\.\d+){2}$', line):
            lumo_line = -index
            lumo = float(data[lumo_line].split()[3])
            break

    # Magnetization
    for line in reversed(data):
        if re.findall('.*N_up - N_down.*', line):
            mag = re.search('.*N_up - N_down.*', line).string.split()[7]
            break

    # Total energy...
    for line in reversed(data):
        if re.findall('.*Total energy, T -> 0.*', line):
            energy_tot = float(re.search('.*Total energy, T -> 0.*',
                                         line).string.split()[9])
            break

    # xyz and chemical symbols
    for line in data:
        if re.findall('.*Number of atoms.*', line):
            number_of_atoms = int(re.search('.*Number of atoms.*',
                                            line).string.split()[5])
            break
    xyzfound = False
    for index, line in enumerate(reversed(data)):
        if re.findall('.*Final atomic structure:.*', line):
            first_atom_line = -index+1
            xyzfound = True
            break
    if not xyzfound:
        for index, line in enumerate(data):
            if re.findall('Updated atomic structure:', line):
                print('error')
        for index, line in enumerate(data):
            if re.findall('^\s+atom(\s+(|-)\d\.\d+){3}\s+\w+', line):
                first_atom_line = index
                break

    positions = []
    chemical_elements = []
    for line in data[first_atom_line:first_atom_line+number_of_atoms]:
        if re.findall('^\s+atom\s+((-|)\d+\.\d+).*', line):
            chemical_elements.append(np.array(
                re.search('^\s+atom\s+((-|)\d+\.\d+).*',
                          line).string.split()[4], dtype=str))
            positions.append(np.array(
                re.search('^\s+atom\s+((-|)\d+\.\d+).*',
                          line).string.split()[1:4], dtype=float))
    positions = np.array(positions)
    chemical_elementsaux = np.array(chemical_elements)
    cebag = np.array([ce for ce in chemical_elementsaux])

    # Chamical formula
    chemical_species, chemical_species_qtn = np.unique(chemical_elements,
                                                       return_counts=True)
    list_symb_qtn_tuple = sorted(zip(chemical_species, chemical_species_qtn))
    nparray_symb_qtn_tuple = np.array(list(map(list, list_symb_qtn_tuple)))
    chemical_formula = ''.join([''.join(i) for i in nparray_symb_qtn_tuple])

    # returning data
    return gap, homo, lumo, mag, energy_tot, positions, cebag, chemical_formula


def fromfolders_extract_fhi(folders, output_file_name='aims.out'):
                          
    """It get data from fhi calculation folders.
    Parameters
    ----------
    folders: np.array
             A numpy array with the names of the folders with calculations.
    output_file_name: str, (optional, default='aims.out')
                      The output file name.
    Return
    ------
    newdata: np.array with coluns for each new feature.
    """

    # The lists list_of_new_features_data and list_of_new_features_name, colect
    # All the information along the analysis
    list_of_new_features_data = []
    list_of_new_features_name = []

    # getting properties from  extractor_fhi
    gap_list = []
    homo_list = []
    lumo_list = []
    mag_list = []
    etot_list = []
    positions_list = []
    cheme_list = []
    chemf_list = []
    folders = np.array(folders.values)
    print("Initializing fromfolders_extract_fhi:")
    for index in range(len(folders)):
        folder = folders[index]
        gap, homo, lumo, mag, etot, positions, cheme, chemf = extractor_fhi(
            folder + '/' + output_file_name)
        if index % 50 == 0:
            print("    concluded %3.1f%%" % (100*index/len(folders)))
        positions = positions.tolist()
        cheme = cheme.tolist()
        gap_list.append(gap)
        homo_list.append(homo)
        lumo_list.append(lumo)
        mag_list.append(mag)
        etot_list.append(etot)
        positions_list.append(positions)
        cheme_list.append(cheme)
        chemf_list.append(chemf)
    print("    concluded %3.1f%%" % (100))
    # Adding new features data and name to its lists
    list_of_new_features_data = [gap_list, homo_list, lumo_list, mag_list,
                                 etot_list, positions_list, cheme_list,
                                 chemf_list]
    list_of_new_features_name = ['reg_gap', 'reg_homo', 'reg_lumo', 'reg_mag',
                                 'reg_etot', 'bag_positions', 'bag_cheme', 
                                 'reg_chemf']

    # Getting for each sample the quantity of each specie in the dataset
    jointedformulas = ''.join(chemf_list)
    allspeciesrepeated = re.findall('[abcdefghiklmnopqrstuvxywzABCDEFGHIKLMNOPQRSTUVXYWZ]+',
                                    jointedformulas)
    allspecies_list = np.unique(allspeciesrepeated).tolist()
    list_features_qtn_atoms_data = []
    list_features_qtn_atoms_name = []

    for specie in allspecies_list:
        list_features_qtn_atoms_name.append('reg_qtn_' + specie)
        list_of_qnt = []
        for formula in chemf_list:
            specieplusnumber = re.findall(specie+'\d+', formula)
            if specieplusnumber:
                number = re.findall('\d+', specieplusnumber[0])[0]
            else:
                number = 0
            list_of_qnt.append(int(number))
        list_features_qtn_atoms_data.append(list_of_qnt)
    # Adding new features data and name to its lists

    for new_feature_data in list_features_qtn_atoms_data:
        list_of_new_features_data.append(new_feature_data)
    for new_feature_name in list_features_qtn_atoms_name:
        list_of_new_features_name.append(new_feature_name)

    return list_of_new_features_name, list_of_new_features_data


def extractor_vasp(outfile='OUTCAR'):
    """This function open a VASP output file (outfile), read it, than and return the
    important informations.
    Parameters
    ----------
    outfile: string (optional, default='OUTCAR')
            The name of the output file.
    Return
    ------
    gap, homo, lumo, mag, energy_tot: floats.
                                  Information of gap, HOMO, LUMO,
                                  magnetization, and total energy,
                                  respectively.
    positions: np.array of floats, (n,3) shaped.
               The cartezian positions of the atoms.
    cheme: np.array of floats, lengh n.
           The chemical species of the structure.
    chemf: str.
           String with the chemical species in alphabetical order
           followed by the quantity of atoms.
    """

    # The test of the input variables are necessary.

    with open(outfile, mode='r') as file:
        data = file.readlines()

    # eigenstates
    for line in data:
        if re.findall('number of bands    NBANDS', line):
            nkp= int(line.split()[3])
            nbands= int(line.split()[-1])
            break
    eigen = np.zeros([2,nkp,nbands])
    occup = np.zeros([2,nkp,nbands])
    for n, line in enumerate(reversed(data)):
        if re.findall('spin component 1', line):
            for line in data[-n-1:]:
                if re.findall('spin component', line):
                    spin = int(line.split()[2]) - 1
                if re.findall('k-point', line):
                    kp = int(line.split()[1]) - 1
                if re.findall(r'^\s+\d+', line):
                    band, energy, occ = line.split()
                    eigen[spin,kp,int(band)-1] = energy
                    occup[spin,kp,int(band)-1] = occ
                if '------' in line: break
            break
    cbm = max(eigen[occup > 0.5])
    vbm = min(eigen[occup < 0.5])
    gap = vbm - cbm

    # charges
    for n, line in enumerate(reversed(data)):
        if re.findall('total charge', line):
            chgst = []
            chgss = []
            chgsp = []
            chgsd = []
            chgsf = []
            for i in range(-n+3, 0):
                if '---' in data[i]: break
                aux = data[i].split()
                chgss.append(aux[1])  # for LORBIT  = 10
                chgsp.append(aux[2])  # for LORBIT  = 10
                chgsd.append(aux[3])  # for LORBIT  = 10
                chgsf.append(aux[4])  # for LORBIT  = 10
                chgst.append(aux[5])  # for LORBIT  = 10
            break
    chgst = np.array(chgst, dtype=float).tolist()
    chgss = np.array(chgss, dtype=float).tolist()
    chgsp = np.array(chgsp, dtype=float).tolist()
    chgsd = np.array(chgsd, dtype=float).tolist()
    chgsf = np.array(chgsf, dtype=float).tolist()

    # Magnetization
    for n, line in enumerate(reversed(data)):
        if re.findall('magnetization', line):
            magst = []
            magss = []
            magsp = []
            magsd = []
            magsf = []
            for i in range(-n+3, 0):
                if '---' in data[i]: break
                aux = data[i].split()
                magss.append(aux[1])  # for LORBIT  = 10
                magsp.append(aux[2])  # for LORBIT  = 10
                magsd.append(aux[3])  # for LORBIT  = 10
                magsf.append(aux[4])  # for LORBIT  = 10
                magst.append(aux[5])  # for LORBIT  = 10
            break
    magst = np.array(magst, dtype=float).tolist()
    magss = np.array(magss, dtype=float).tolist()
    magsp = np.array(magsp, dtype=float).tolist()
    magsd = np.array(magsd, dtype=float).tolist()
    magsf = np.array(magsf, dtype=float).tolist()

    # cell
    for n,line in enumerate(reversed(data)):
        if re.findall('direct lattice vectors', line):
            cell = np.array([data[-n].split()[0:3],
                             data[-n+1].split()[0:3],
                             data[-n+2].split()[0:3]], dtype=float
                            ).tolist()
            break

    # Total energy
    for line in reversed(data):
        if re.findall('energy  without entropy=     ', line):
            energy = float(line.split()[6])
            break

    # mag
    for line in reversed(data):
        if re.findall(r'^ number of electron\s+ \d+\.\d+\s+magnetization', line):
            mag = float(line.split()[5])
            break

    # efermi
    for line in reversed(data):
        if re.findall('E-fermi', line):
            efermi = float(line.split()[2])
            break

    # positions
    for n, line in enumerate(reversed(data)):
        if re.findall('POSITION', line):
            positions = []
            for i in range(-n+1, 0):
                if '---' in data[i]:
                    break
                positions.append(data[i].split()[0:3])
            break
    positions = np.array(positions, dtype=float)
    # Comment the line bellow to tune the speed
    force_center = True
    if force_center == True:
        from sklearn.cluster import DBSCAN
        clust = DBSCAN(eps=3, min_samples=1).fit(positions)  # 3 angstroms
        while max(clust.labels_) > 0:
            aux = np.sum(np.array(cell)/2, axis=0) - np.average(positions[clust.labels_ == 0], axis=0)
            cart_to_direct = np.linalg.inv(cell).T
            direct_poss = np.dot(cart_to_direct, positions.T).T
            aux_direct = np.dot(cart_to_direct, aux).T
            direct_new_pos = direct_poss + aux_direct
            aux2 = direct_new_pos.flatten()
            for i, val in enumerate(aux2):
                if val < 0: aux2[i] = val + 1
                if val > 1: aux2[i] = val - 1
            direct_new_pos = aux2.reshape(-1, 3)
            new_pos = np.dot(cell, direct_new_pos.T).T
            positions = new_pos.tolist()
            clust = DBSCAN(eps=3, min_samples=1).fit(positions)

    # cheme
    potcars = []
    for n, line in enumerate(data):
        if re.findall('POTCAR:', data[n]):
            name = line.split()[2]
            if '_' in name:
                name = name.split('_')[0]
            potcars.append(name)
        if re.findall('POSCAR:', line): break
    potcars = potcars[0:int(len(potcars)/2)]
    speciesquantities = []
    for n, line in enumerate(data):
        if re.findall('ions per type =', data[n]):
            speciesquantities = np.array(line.split()[4:], dtype=int)
            break
    cheme = []
    for index, pot in enumerate(potcars):
        for i in range(speciesquantities[index]):
            cheme.append(pot)

    # chemf
    chemf_parts = [str(symb) + str(qtn) for qtn, symb in zip(
        speciesquantities, potcars)]
    chemf_parts = np.array(chemf_parts)
    chemf_parts.sort()
    chemf = ''.join(chemf_parts)

    # returning data
    return cbm, vbm, gap, mag, energy, positions, cell, cheme, chemf, \
           chgss, chgsp, chgsd, chgsf, chgst, magst, magss, magsp, \
           magsd, magsf


def fromfolders_extract_vasp(folders, output_file_name='OUTCAR'):
    """It get data from vasp calculation folders.
    Parameters
    ----------
    folders: np.array
             A numpy array with the names of the folders with calculations.
    output_file_name: str, (optional, default='OUTCAR')
                      The output file name.

    Return
    ------
    list_of_new_features_name: list of str.
                               New data name.

    list_of_new_features_data: list of np.array.
                               New data.
    """

    cbm_list = []
    vbm_list = []
    gap_list = []
    mag_list = []
    energy_list = []
    positions_list = []
    cell_list = []
    cheme_list = []
    chemf_list = []
    chgss_list = []
    chgsp_list = []
    chgsd_list = []
    chgsf_list = []
    chgst_list = []
    magst_list = []
    magss_list = []
    magsp_list = []
    magsd_list = []
    magsf_list = []
    folders = np.array(folders.values)
    print("Initializing fromfolders_extract_vasp:")
    for index, folder in enumerate(folders):
        data = extractor_vasp(folder + '/' + output_file_name)
        cbm, vbm, gap, mag, energy, positions, cell, cheme, chemf, chgss, \
            chgsp, chgsd, chgsf, chgst, magst, magss, magsp, magsd, magsf = data
        if index % 50 == 0:
            print("    concluded %3.1f%%" % (100*index/len(folders)))
        cbm_list.append(cbm)
        vbm_list.append(vbm)
        gap_list.append(gap)
        mag_list.append(mag)
        energy_list.append(energy)
        positions_list.append(positions)
        cell_list.append(cell)
        cheme_list.append(cheme)
        chemf_list.append(chemf)
        chgss_list.append(chgss)
        chgsp_list.append(chgsp)
        chgsd_list.append(chgsd)
        chgsf_list.append(chgsf)
        chgst_list.append(chgst)
        magst_list.append(magst)
        magss_list.append(magss)
        magsp_list.append(magsp)
        magsd_list.append(magsd)
        magsf_list.append(magsf)
    print("    concluded %3.1f%%" % (100))

    list_of_new_features_data = [cbm_list, vbm_list, gap_list, mag_list,
                                 energy_list, positions_list, cell_list, 
                                 cheme_list, chemf_list, chgss_list, 
                                 chgsp_list, chgsd_list, chgsf_list, 
                                 chgst_list, magst_list, magss_list, 
                                 magsp_list, magsd_list, magsf_list]
    list_of_new_features_name = ['reg_cbm', 'reg_vbm', 'reg_gap', 'reg_mag',
                                 'reg_energy', 'bag_positions', 'bag_cell', 
                                 'bag_cheme', 'reg_chemf', 'bag_chgss', 
                                 'bag_chgsp', 'bag_chgsd', 'bag_chgsf', 
                                 'bag_chgst', 'bag_magst', 'bag_magss', 
                                 'bag_magsp', 'bag_magsd', 'bag_magsf']:

    return list_of_new_features_name, list_of_new_features_data
