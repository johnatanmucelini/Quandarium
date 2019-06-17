"""Several tools to deal with QC calculations."""

def find(code, path_to_start_the_search='.', recursivity=True ):
    """This function find the path of all folders with calculations.
    In fact it verify if the output file for the selected code is in
    the folder. If recursivity == False, the search run over all the
    folders inside the path_to_start_the_search, otherwise the seach
    run recursively."""

    import os.path.isfile
    import logging

    logging.basicConfig(level=logging.DEBUG)
    logging.info('The logging level is debug')

    # Verification
    if os.path.isdir(path_to_start_the_search) == False :
        logging.error('The key path_to_start_the_search (',
                       str(path_to_start_the_search),
                       ') is not a directory!')
    else:
        logging.debug('path_to_start_the_search is a directory!')

    # Geting the output_file_name from the code input var
    if code == 'VASP': output_file_name = 'OUTCAR'
    if code == 'fhi': output_file_name = 'aims.out'
    if code not in ['VASP', 'fhi'] :
        logging.info( 'The code variable (', str(code), ') is not avaliable.',
                      'This variable will be used as a output file name.'
        output_file_name = code
    logging.debug('The output_file_name variable ware selected to '
                   + output_file_name + '.' )

    # Find the list with paths
    if recursivity == True:
        list_of_folders = [ x[0] for x in
                            os.walk(path_to_start_the_search + '/') ]
    if recursivity == False:
        list_of_folders = [ x for x in os.listdir(path_to_start_the_search +'/')
                             if os.path.isdir(path_to_start_the_search + '/' + x) ]
    logging.debug('list_of_folders were find, it present len of'
                  + str(len(list_of_folders)) + '.')

    # Test
    if len(list_of_folders) == 0 :
        logging.error('No one folder ware found.')

    # Finding the folders with calculations
    list_with_all_calculations_folder = []
    for folder in list_of_folders :
        if os.path.isfile( folder + '/' + output_file_name ) :
            list_with_all_calculations_folder.append(folder)

    return list_with_all_calculations_folder

def get_folders( list_with_all_calculations_folder )
