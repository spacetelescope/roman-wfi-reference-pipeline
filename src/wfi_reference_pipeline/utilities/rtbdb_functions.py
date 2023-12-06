import logging
from rtb_db.utilities import rfp_tools, login


def get_ma_table_from_rtbdb(ma_table_number=None):
    """
    This method get_ma_table_from_rtb_db() accesses the rtb_database information on MA tables.

    Parameters
    ----------
    ma_table_number: integer
        Data base entry for unique MA Table number.

    Returns
    -------
    ma_table_dict: dictionary
        A python dictionary of MA Table information. See RTB Database examples for keys and values.
    """

    eng = login.connect_server(dsn_name='DWRINSDB')
    ma_table_dict = rfp_tools.query_ma_table(eng, ma_table_number)
    if ma_table_dict:
        logging.info('Successfully read MA Table information from the RTB Database.')
    return ma_table_dict


def make_even_spacing_read_pattern(num_resultants=None, num_rds_per_res=None):
    """
    The method make_even_spacing_read_pattern() is a helper function to generate a read pattern
    for MA Tables with evenly spaced resultants. This is needed because the current MA Tables in
    the RTB Database do not have the same specifications to replace files on CRDS that are used
    for build and regression testing.

    Parameters
    ----------
    num_resultants: integer; default=None
        Integer number of resultants in exposure.
    num_rds_per_res: integer; default=None
        Integer number of reads per resultant for evenly spaced averaged resultants.

    Returns
    -------
    read_pattern: list of lists
        Nested list of lists that are taken created from existing meta data or accessed by new meta roman data models.
    """

    rds_list = list(range(1, num_resultants*num_rds_per_res+1))
    # Make nested list of lists read_pattern for evenly spaced resultants according to DRM, DMS, or GSFC.
    read_pattern = [rds_list[i:i+num_resultants] for i in range(0, len(rds_list), num_resultants)]

    return read_pattern


def write_metrics_db_dark(dark_file_dict, dark_dq_dict, dark_struc_dict, dark_amp_dict):
    """
    This method within the RFP project writes the predetermined multiple dark dictionaries
    of metrics generated within the Dark() module.

    dark_file_dict: dict, default=None
        This is the file metric dictionary with dates, etc.
    dark_dq_dict: dict, default=None
        This is the dq dictionary determined by reftype flags, etc.
    dark_struc_dict dict, default=None
        This is the structure dictionary containing 2D spatial information of the detector.
    dark_amp_dict: dict, default=None
        This is the amplifier dictionary containing various metrics of sections of pixels according to amplifier.
    """

    eng = login.connect_server(dsn_name='DWRINSDB')
    rfp_tools.add_new_dark_file(eng, dark_file_dict, dark_dq_dict,dark_struc_dict, dark_amp_dict)
