import logging
from datetime import datetime

import romancal

try:
    from rtb_db.constants.rfp_reef import DB_QC_INCOMPLETE
    from rtb_db.table_defs.wfi_rfp.log import RFPLogProTable
except ImportError:
    logging.warning("Attempting to import rtb_db when not available, install package using rtb_db optional dependency")


class DBEntry:
    """
    Database Class to consolidate all potential table classes.
    This class is designed to be accessed from the datbase_handler utility.
    """

    def __init__(self):
        self.rfp_log_pro = None # DB entry for Logistics Processing Table

    @staticmethod
    def get_date_time_formatted():
        return datetime.today().replace(microsecond=0)

    def init_rfp_log_pro(self, ref_type, wfi_mode, reef_monitor):
        """
        Initialize a logistics processing table object.

        Parameters
        ----------
        ref_type : str
            Reference type associated with this pipeline run.
        wfi_mode : str
            wfi mode associated with this pipeline run.
        reef_monitor : bool
            Expecting external monitoring for this run.
        """
        start_time = self.get_date_time_formatted()
        self.rfp_log_pro = RFPLogProTable(ref_type=ref_type,
                                          start_time=start_time,
                                          wfi_mode=wfi_mode,
                                          reef_monitor=reef_monitor,
                                          rcal_version=romancal.__version__,
                                          rfp_version="0.0.1", # TODO GET PROPER VERSION
                                          qc_status=DB_QC_INCOMPLETE,
                                          )

# NOTE The following will need to eventually be added to the RFPLogProTable during the course of the pipeline run
#   pipeline_cmd:           Mapped[str] = mapped_column(String())
#   prep_start:         Mapped[datetime] = mapped_column(DateTime())
#   prep_end:           Mapped[datetime] = mapped_column(DateTime())
#   pipe_start:         Mapped[datetime] = mapped_column(DateTime())
#   pipe_end:           Mapped[datetime] = mapped_column(DateTime())
#   output_filename
#   end_time:           Mapped[datetime] = mapped_column(END_DATETIME_COL, DateTime())
#   qc_status:          Mapped[str] = mapped_column(QC_STATUS_COL, String()) <---- Currently in but UPDATE THIS
#   crds_filename:      Mapped[Optional[str]] = mapped_column(CRDS_FILENAME_COL, String())
#   crds_context:       Mapped[str] = mapped_column(CRDS_CONTEXT_COL, String())
#   crds_end_time:      Mapped[Optional[datetime]] = mapped_column(DateTime())
#   crds_start_time:    Mapped[Optional[datetime]] = mapped_column(DateTime())
#   crds_delivered:     Mapped[bool] = mapped_column(CRDS_DELIVERED_COL, Boolean())
#   _input_file_list:   Mapped[str] = mapped_column(String())
#   input_file_list = String2ListVariable()