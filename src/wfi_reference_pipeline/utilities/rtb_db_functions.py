import numpy as np
import wfi_reference_pipeline.resources.data as resource_meta
import psutil, sys, os, glob, time, gc, asdf, math, datetime, logging, yaml, importlib.resources
from pathlib import Path
from RTB_Database.utilities.login import connect_server
from RTB_Database.utilities.table_tools import DatabaseTable

# Load all of the yaml files with reference file specific meta data
meta_yml_fls = importlib.resources.files(resource_meta)

anc_file_path = Path("/grp/roman/rcosenti/RFP_git_clone/wfi_reference_pipeline/src/wfi_reference_pipeline/" 
                     "resources/data/ancillary.yaml")

# Load the YAML file contents into a dictionary using safe_load()
with anc_file_path.open() as af:
    anc_data = yaml.safe_load(af)


def get_ma_tab_from_rtb_db(ma_table_id):
    """
    This method get_me_table_info() imports modules and methods from the RTB database
    repo to allow the RFP to establish a connection and query the science ma tables
    for specifications on how they are made.

    NOTE: Incorporating changes from the upscope are not included yet and will alter
    how this is done in significant manners.

    It might be that this is also a utility in order to initiliaze dark reference file
    meta data with ma table information before the Dark class() instance is created

    Parameters
    ----------
    ma_table_id: integer
        Database entry in first column to fetch ma table information.

    Returns
    -------
    ma_tab_num_resultants: integer
        number of resultants, currently called ngroups in meta
    ma_tab_reads_per_resultant: integer
        number of reads per resultant, currently called nframe in meta
    """

    ma_tab_meta = {}

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
    ma_tab_meta['exposure'].update(dict(ngroups=ma_tab_num_resultants, nframes=ma_tab_reads_per_resultant, groupgap=0,
                                        ma_table_name=ma_tab_name, ma_table_number=ma_table_id))
    logging.info(f'Updated meta data with MA table info.')
    # Determine WIM or WSM - WFI Imaging or Spectral Mode - from the ma table specs in the RTB database from PRD.
    if frame_time == anc_data['frame_time']['WIM']:  # frame time in imaging mode in seconds
        ma_tab_meta['exposure'].update({'type': 'WFI_IMAGE', 'p_exptype': 'WFI_IMAGE|'})
    elif frame_time == anc_data['frame_time']['WSM']:  # frame time in spectral mode in seconds:
        ma_tab_meta['exposure'].update({'type': 'WFI_GRISM', 'p_exptype': 'WFI_GRISM|WFI_PRISM|'})

    ma_tab_sequence = '222222R'

    return ma_tab_meta, ma_tab_num_resultants, ma_tab_reads_per_resultant, ma_tab_sequence
