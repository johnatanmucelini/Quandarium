import numpy as np
import re


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
    """
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
    chemical_species = []
    for line in data[first_atom_line:first_atom_line+number_of_atoms]:
        if re.findall('^\s+atom\s+((-|)\d+\.\d+).*', line):
            chemical_species.append(np.array(
                re.search('^\s+atom\s+((-|)\d+\.\d+).*',
                          line).string.split()[4], dtype=str))
            positions.append(np.array(
                re.search('^\s+atom\s+((-|)\d+\.\d+).*',
                          line).string.split()[1:4], dtype=float))
    positions = np.array(positions)
    chemical_species = np.array(chemical_species)

    # returning data
    return gap, homo, lumo, mag, energy_tot, positions, chemical_species
