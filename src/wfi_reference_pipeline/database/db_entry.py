from datetime import datetime
from importlib.metadata import version

import romancal
from rtb_db.constants.rfp_reef import FORMAT_DATE_TIME
from rtb_db.table_defs.wfi_rfp.log import RFPLogProTable


class DBEntry:
    """
    Database Class to consolidate all potential table classes.
    This class is designed to be accessed from the datbase_handler utility.
    """

    def __init__(self):
        self.rfp_log_pro = None # DB entry for Logistics Processing Table

    def init_rfp_log_pro(self, ref_type, wfi_mode, reef_monitor):

        current_time = datetime.now()
        start_time = current_time.strftime(FORMAT_DATE_TIME)
        __version__ = version(__package__ or "wfi_reference_pipeline")

        self.rfp_log_pro = RFPLogProTable(ref_type=ref_type,
                                          start_time=start_time,
                                          wfi_mode=wfi_mode,
                                          reef_monitor=reef_monitor,
                                          rcal_version=romancal.__version__,
                                          rfp_version=__version__)

        ## SAPP TODO -
        # ITEMS NOT NEEDED AT INITIALIZATION:
        # pipeline:           Mapped[str] = mapped_column(String()) # THIS ONE MAY NOT BE NEEDED
        # prep_start:         Mapped[datetime] = mapped_column(DateTime())
        # prep_end:           Mapped[datetime] = mapped_column(DateTime())
        # pipe_start:         Mapped[datetime] = mapped_column(DateTime())
        # pipe_end:           Mapped[datetime] = mapped_column(DateTime())
        # output_filename
        # end_time:           Mapped[datetime] = mapped_column(END_DATETIME_COL, DateTime())
        # qc_status:          Mapped[str] = mapped_column(QC_STATUS_COL, String())
        # crds_filename:      Mapped[Optional[str]] = mapped_column(CRDS_FILENAME_COL, String())
        # crds_context:       Mapped[str] = mapped_column(CRDS_CONTEXT_COL, String())
        # crds_end_time:      Mapped[Optional[datetime]] = mapped_column(DateTime())
        # crds_start_time:    Mapped[Optional[datetime]] = mapped_column(DateTime())
        # crds_delivered:     Mapped[bool] = mapped_column(CRDS_DELIVERED_COL, Boolean())
        # _input_file_list:   Mapped[str] = mapped_column(String())
        # input_file_list = String2ListVariable()