"""Several tools to deal with QC calculations."""

import sys
import os.path
from distutils.dir_util import copy_tree


def find(code_or_file, path_to_start_the_search='.', recursivity=True):
    """This function find the path of all folders with calculations.
    In fact, if code_or_file is a code it select folder with the code output
    file, if code_or_file is not a code (so it is a filename), it select
    folders with a file named equal to code_or_file.

    Parameters
    ----------
    code_or_file: str.
                  The code name (VASP of fhi) or a file name to search to
                  consider that the folder with the filename is a calculation
                  folder.
    path_to_start_the_search: str, optional.
                              path_to_start_the_search is the path to start the
                              seaching for folders with calculations.
    recursivity: boolean
                 If True, the search is completely recursive. If False, the
                 search run over the forder within the folder in the variable
                 path_to_start_the_search.

    Returns
    -------
    list_with_all_calculations_folder: list of size equal to the number of
                                       The path to the folder with the
                                       calculations.

    Examples
    --------
    >>> import pandas as pd
    >>> from quandarium.calculations import find
    >>>
    >>> dataframe = pd.DataFrame()
    >>> dataframe['calcfolder'] = find('VASP', '.')
    """

    # Variables unit tests
    if not isinstance(code_or_file, str):
        print("code_or_file must be a string! Aborting...")
        sys.exit(1)

    if not isinstance(path_to_start_the_search, str):
        print("path_to_start_the_search must be a string! Aborting...")
        sys.exit(1)

    if not isinstance(recursivity, bool):
        print("recursivity must be an bool! Aborting...")
        sys.exit(1)

    # Other verification tests
    if not os.path.isdir(path_to_start_the_search):
        print("The path_to_start_the_search is not a directory! Aborting...")
        sys.exit(1)

    # Geting the output_file_name from the code input var
    if code_or_file == 'VASP':
        output_file_name = 'OUTCAR'
    if code_or_file == 'fhi':
        output_file_name = 'aims.out'
    if code_or_file not in ['VASP', 'fhi']:
        print('{} variable is not an avaliable code.'.format(code_or_file))
        print('code_or_file will be considered as the output file name.')
        output_file_name = code_or_file
    print('The output_file_name variable were selected to {}.'.format(
        output_file_name))

    # Find the list with paths
    if recursivity:
        list_of_folders = [x[0] for x in os.walk(path_to_start_the_search
                                                 + '/')]
    if not recursivity:
        list_of_folders = [x for x in os.listdir(path_to_start_the_search +
                                                 '/')
                           if os.path.isdir(path_to_start_the_search + '/' +
                                            x)]
    print('list_of_folders were find, it present len of {}.'.format(
        len(list_of_folders)))

    # Test
    if len(list_of_folders) == 0:
        print('No one folder were found! Aborting...')
        sys.exit(1)

    # Finding the folders with calculations
    list_with_all_calculations_folder = []
    for folder in list_of_folders:
        if os.path.isfile(folder + '/' + output_file_name):
            list_with_all_calculations_folder.append(folder)

    # Test
    if len(list_with_all_calculations_folder) == 0:
        print('No one folder were found with calculations!')
    else:
        # Printing folders with calculations
        print('The following folders present calculation:')
        for folder in list_with_all_calculations_folder:
            print('    {}'.format(folder))

    print('Folders founded:')
    for i in list_with_all_calculations_folder:
        print('    {}'.format(i))

    list_with_all_calculations_folder.sort()

    return list_with_all_calculations_folder


def cp_folders(list_with_all_calculations_folder, final_folder):
    """This function copy the each folder in the
    list_with_all_calculations_folder to the final_folder.

    Arguments:
    list_with_all_calculations_folder: list of stings with the path to the
                                       folders.
    final_folder: path to the final folder. If it does not exist, it will be
                  created

    Example:
    >>> cp_folders(list_of_calculations_folders, './all_calculations')
    """

    print('Initializing cp_folders function!')

    # Need to verify the input variables!

    for calculation_folder in list_with_all_calculations_folder:
        copy_tree(calculation_folder, final_folder)

    print('All the folders were copied')
