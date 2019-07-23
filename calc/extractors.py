"""This module present functions that extract data from Quantum Chemistry (QC)
Calculations, and related functions. The QC codes with extractors avaliable
are:
- FHI-aims
- VESTA (not yet)
"""

import logging
import re
import pandas as pd
import numpy as np
from quandarium.analy.aux import arr2bag
from quandarium.analy.aux import bag2arr

logging.basicConfig(filename='/home/johnatan/quandarium_module.log',
                    level=logging.INFO)
logging.info('The logging level is INFO')


def extractor_fhi(outfile='aims.out'):
    """This function open a fhi output file and read and return the important
    informations.
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
    for index, line in enumerate(reversed(data)):
        if re.findall('.*Final atomic structure:.*', line):
            first_atom_line = -index+1
            break
    for line in data:
        if re.findall('.*Number of atoms.*', line):
            number_of_atoms = int(re.search('.*Number of atoms.*',
                                            line).string.split()[5])
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
    cebag = np.array(["\"" + ce + "\"" for ce in chemical_elementsaux])

    # Chamical formula
    chemical_species, chemical_species_qtn = np.unique(chemical_elements,
                                                       return_counts=True)
    list_symb_qtn_tuple = sorted(zip(chemical_species, chemical_species_qtn))
    nparray_symb_qtn_tuple = np.array(list(map(list, list_symb_qtn_tuple)))
    chemical_formula = ''.join([''.join(i) for i in nparray_symb_qtn_tuple])

    # returning data
    return gap, homo, lumo, mag, energy_tot, positions, cebag, chemical_formula


def extractfhi_fromfolders(folders, output_file_name='aims.out', csv_name=''):
    """It get data from fhi calculation folders.
    Parameters
    ----------
    folders: list of str.
             List with all the fhi calculation paths.
    output_file_name: str, (optional, default='aims.out')
                      The output file name.
    csv_name: str, optional (default='')
              If provided, csv file with this name will be writed.
    Return
    ------
    dataset: pd.DataFrame
             A pandas data frame with all the information.
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
    print("Initializing data extraction:")
    for index, folder in enumerate(folders):
        gap, homo, lumo, mag, etot, positions, cheme, chemf = extractor_fhi(
            folder + '/aims.out')
        if index % 50 == 0:
            print("    concluded %3.1f%%" % (100*index/len(folders)))
        positions_bag = arr2bag(positions)
        cheme_bag = arr2bag(cheme)
        gap_list.append(gap)
        homo_list.append(homo)
        lumo_list.append(lumo)
        mag_list.append(mag)
        etot_list.append(etot)
        positions_list.append(positions_bag)
        cheme_list.append(cheme_bag)
        chemf_list.append(chemf)
    print("    concluded %3.1f%%" % (100))
    # Adding new features data and name to its lists
    for new_feature_data in [gap_list, homo_list, lumo_list, mag_list,
                             etot_list, positions_list, cheme_list,
                             chemf_list]:
        list_of_new_features_data.append(new_feature_data)
    for new_feature_name in ['reg_gap', 'reg_homo', 'reg_lumo', 'reg_mag',
                             'reg_etot', 'bag_positions',
                             'bag_cheme', 'reg_chemf']:
        list_of_new_features_name.append(new_feature_name)

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

    # calculation folder
    calcfolder = []
    for folder in folders:
        calcfolder.append(folder.split('/')[-1])
    # Adding new features data and name to its lists
    list_of_new_features_data.append(calcfolder)
    list_of_new_features_name.append('calcfolder')

    # Creating the pandas DataFrame
    new_dataframe = pd.DataFrame(np.array(list_of_new_features_data).T,
                                 columns=list_of_new_features_name)

    if csv_name:
        new_dataframe.to_csv('new_dataframe.csv', sep=' ', index=False)

    return new_dataframe
