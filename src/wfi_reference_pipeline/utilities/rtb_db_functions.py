import wfi_reference_pipeline.resources.data as resource_meta
import logging, yaml, importlib.resources
#from RTB_Database.utilities.login import connect_server
#from RTB_Database.utilities.table_tools import DatabaseTable

# File to empty dictionary of common meta keys.
meta_yaml_files = importlib.resources.files(resource_meta)

# Load the YAML file contents into a dictionary using safe_load().
anc_yaml_path = meta_yaml_files.joinpath('ancillary.yaml')

# Load the YAML file contents into a dictionary using safe_load()
with anc_yaml_path.open() as ayp:
    anc_data = yaml.safe_load(ayp)


def get_ma_table_from_rtb_db(ma_table_id):
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

    # Determine WIM or WSM - WFI Imaging or Spectral Mode - from the ma table specs in the RTB database from PRD.
    if frame_time == anc_data['frame_time']['WIM']:  # frame time in imaging mode in seconds
        ma_table_meta['exposure'].update({'type': 'WFI_IMAGE', 'p_exptype': 'WFI_IMAGE|'})
    elif frame_time == anc_data['frame_time']['WSM']:  # frame time in spectral mode in seconds:
        ma_table_meta['exposure'].update({'type': 'WFI_GRISM', 'p_exptype': 'WFI_GRISM|WFI_PRISM|'})
    logging.info(f'Updated meta data with MA table info.')

    return ma_table_meta


def make_read_pattern(meta=None, num_resultants=None, num_rds_per_res=None, even_spacing=True):
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
    ma_table_meta: dictionary; default=None
        Dictionary with necessary meta data to make MA table read pattern
    num_resultants: integer; default=None
        Integer number of resultants in exposure.
    num_rds_per_res: integer; default=None
        Integer number of reads per resultant for evenly spaced averaged resultants.
    even_spacing: keyword; default=True
        Assuming evenly spaced resultants until read pattern is included in meta data, at which time this might be
        determined to be unnecessary.

    Returns
    -------
    read_pattern: list of lists
        Nested list of lists that are taken created from existing meta data or accessed by new meta roman data models.
    """

    if even_spacing:
        if meta:
            num_resultants = meta['exposure']['ngroups']
            num_rds_per_res = meta['exposure']['nframes']
            rds_list = list(range(1, num_resultants*num_rds_per_res+1))
            # Make nested list of lists read_pattern for evenly spaced resultants according to DRM, DMS, or GSFC.
            read_pattern = [rds_list[i:i+num_resultants] for i in range(0, len(rds_list), num_resultants)]
        else:
            rds_list = list(range(1, num_resultants*num_rds_per_res+1))
            # Make nested list of lists read_pattern for evenly spaced resultants according to DRM, DMS, or GSFC.
            read_pattern = [rds_list[i:i+num_rds_per_res] for i in range(0, len(rds_list), num_rds_per_res)]
    else:
        # Default unevenly spaced MA table sequence for RFP development and testing.
        read_pattern = [[1,2], [4,5,6], [9, 10, 11,12], [13]]

    return read_pattern
