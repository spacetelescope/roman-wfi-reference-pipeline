import logging, json, tempfile, os
from rtb_db.utilities import rfp_tools, login


def get_ma_table_from_rtbdb(ma_table_id):
    """
    This method get_ma_table_from_rtb_db() accesses the rtb_database repo for methods to connect to the
    the RTB database and access information on MA tables.

    Parameters
    ----------
    ma_table_id: integer
        Database entry in first column to fetch MA table information.

    Returns
    -------
    ma_table_meta: dictionary
        A python dictionary of all meta data provided by the RTB database for MA table instructions
        necessary to populate the exposure key required by the Dark roman data model.
    """

    # connect to database and access MA table information
    con, _, _ = connect_server(DSN_name='DWRINSDB')
    new_tab = DatabaseTable(con, 'ma_table_science')
    ma_tab = new_tab.read_table()
    ma_tab_ind = ma_table_id - 1  # to match index starting at 0 in database with integer ma table ID starting at 1

    ma_tab_name = ma_tab.at[ma_tab_ind, 'ma_table_name']
    ma_tab_reads_per_resultant = ma_tab.at[ma_tab_ind, 'read_frames_per_resultant']
    ma_tab_num_resultants = ma_tab.at[ma_tab_ind, 'resultant_frames_onboard']
    frame_time = ma_tab.at[ma_tab_ind, 'detector_read_time']
    ma_tab_reset_read_time = ma_tab.at[ma_tab_ind, 'detector_reset_read_time']
    logging.info(f'Retrieved RTB Database multi-accumulation (MA) table ID {ma_table_id}.')
    logging.info(f'MA table {ma_tab_name} has {ma_tab_num_resultants} resultants and {ma_tab_reads_per_resultant}'
                 f' reads per resultant.')
    # now update meta data with ma table specs

    ma_table_meta = {}  # initialize empty dictionary for MA table meta
    ma_table_meta['exposure'].update(dict(ngroups=ma_tab_num_resultants, nframes=ma_tab_reads_per_resultant,
                                          groupgap=0, ma_table_name=ma_tab_name, ma_table_number=ma_table_id))

    logging.info(f'Updated meta data with MA table info.')

    return ma_table_meta

def new_method_database():
    """
    from rtb_db.utilities import login
    eng = login.connect_server(DSN_name='DWRINSDB')
    from rtb_db.table_defs.ma_tables import MultiAccumTableScience

    from rtb_db.utilities import table_tools
    print(table_tools.table_names(eng))

    print('--- one row as a class ---\n', table_tools.select_one_from_table(eng, MultiAccumTableScience, MultiAccumTableScience.ma_table_number == 1), "\n")
    ma_tab =  table_tools.select_one_from_table(eng, MultiAccumTableScience, MultiAccumTableScience.ma_table_name == "DEV_TEST")


    ma_tab =  table_tools.select_one_from_table(eng, MultiAccumTableScience, MultiAccumTableScience.ma_table_name == "DEV_TEST")
    ma_tab =  table_tools.select_one_from_table(eng, MultiAccumTableScience, MultiAccumTableScience.ma_table_number == 1)


    :return:
    """
    pass



def make_read_pattern(num_resultants=None, num_rds_per_res=None, uneven_spacing=False):
    """
    The method make_read_pattern is an RFP solution to providing a future meta data field from DMS in which a list
    of lists will be supplied with the information about evenly spaced resultants and the indices of the reads
    averaged together in each resultant. This method does not require access to the RTB database and can be used
    as a standalone function for RFP development and testing mimicking future capabilities of DMS, Roman Attribute
    Dictionary, Roman Data Models, and Romancal.

    example unevenly spaced with skips read_pattern = [[1,2], [4,5,6], [9, 10, 11,12], [13]]
    example evenly spaced no skips read_pattern = [[1,2], [3,4], [5,6], [7,8]]

    Parameters
    ----------
    num_resultants: integer; default=None
        Integer number of resultants in exposure.
    num_rds_per_res: integer; default=None
        Integer number of reads per resultant for evenly spaced averaged resultants.
    uneven_spacing: keyword bool; default=False
        False assuming even spacing until spacing meta data and capabilities are available.
        Return default sample unevenly spaced reads into resultants.

    Returns
    -------
    read_pattern: list of lists
        Nested list of lists that are taken created from existing meta data or accessed by new meta roman data models.
    """

    if uneven_spacing is True:
        # Default unevenly spaced MA table sequence for RFP development and testing.
        read_pattern = [[1, 2], [4, 5, 6], [9, 10, 11, 12], [13]]
    else:
        num_resultants = num_resultants
        num_rds_per_res = num_rds_per_res
        rds_list = list(range(1, num_resultants*num_rds_per_res+1))
        # Make nested list of lists read_pattern for evenly spaced resultants according to DRM, DMS, or GSFC.
        read_pattern = [rds_list[i:i+num_resultants] for i in range(0, len(rds_list), num_resultants)]
    return read_pattern


def write_metrics_db_dark(dark_file_dict, dark_dq_dict, dark_struc_dict, dark_amp_dict):
    """
    This method within the RFP project writes the predetermined multiple dark dictionaries
    of metrics generated within the Dark() module.

    #TODO - Writing a dictionary of dictionaries for all reference file types would be more efficient regardless of reftype
    #TODO - Ingesting the number and names of the dictionaries that are written to tables should be mapped ahead of time where the RFP and RTB-DB both access for read and write

    dark_file_dict: dict, default=None
        This is the file metric dictionary with dates, etc.
    dark_dq_dict: dict, default=None
        This is the dq dictionary determined by reftype flags, etc.
    dark_struc_dict dict, default=None
        This is the structure dictionary containing 2D spatial information of the detector.
    dark_amp_dict: dict, default=None
        This is the amplifier dictionary containing various metrics of sections of pixels according to amplifier.
    """
    eng = login.connect_server(DSN_name='DWRINSDB')

    rfp_tools.add_new_dark_file(eng, dark_file_dict, dark_dq_dict,
                                dark_struc_dict, dark_amp_dict)
