import logging

from rtb_db.table_defs.wfi_rfp.log import RFPLogProTable
from rtb_db.utilities import table_tools
from rtb_db.utilities.login import connect_server
from rtb_db.utilities.table_tools import (
    add_to_tables_from_class_list,
    ensure_connection_is_engine,
)

from wfi_reference_pipeline import constants
from wfi_reference_pipeline.database.db_entry import DBEntry


class DBHandler:
    """
    Utility class to safely handle access to the RTB database.

    This class encapsulates connection setup and basic pipeline entry
    handling against the RTB DB via SQLAlchemy. It is intended to be
    constructed by the Pipeline base class, after configuration has been
    read and validated.
    """

    def __init__(self, ref_type, use_dsn, sql_server_str=None, sql_database_str=None, port=None, dsn_header_str=None):
        """
        Initialize a DBHandler and immediately attempt to connect to the RTB database.

        Parameters
        ----------
        ref_type : str
            Reference type identifier used by the pipeline.
        use_dsn : bool
            If True connect using a DSN name else connect using explicit server/database parameters.
        sql_server_str : str, optional
            SQL Server hostname or address when use_dsn is False.
        sql_database_str : str, optional
            Database name when use_dsn is False.
        port : int, optional
            SQL Server port when use_dsn is False.
        dsn_header_str : str, optional
            DSN name to use when use_dsn is True.

        Attributes
        ----------
        ref_type : str
        sql_id : Any or None
            SQL identifier.
        db_engine : sqlalchemy.engine.Engine or None
            SQLAlchemy Engine once connected.
        db_entry : DBEntry
            Helper object to prepare and insert pipeline entries.
        """


        if ref_type not in constants.WFI_REF_TYPES:
            raise ValueError(
                f"ref_type {ref_type} not valid, must be: {constants.WFI_REF_TYPES}"
            )

        self.ref_type = ref_type
        self.sql_id = None
        self.db_engine = None
        self.db_entry = DBEntry()

        self._connect(use_dsn, sql_server_str, sql_database_str, port, dsn_header_str)


    def _connect(self, use_dsn, sql_server_str, sql_database_str, port, dsn_header_str):
        """
        Establish a connection to the RTB database and store the SQLAlchemy Engine.

        Depending on ``use_dsn``, this will either connect via a named DSN or
        directly using server parameters.

        Parameters
        ----------
        use_dsn : bool
            If True connect using DSN.
        sql_server_str : str, optional
            SQL Server string (non-DSN mode).
        sql_database_str : str, optional
            Database name (non-DSN mode).
        port : int, optional
            SQL Server port (non-DSN mode).
        dsn_header_str : str, optional
            DSN name (DSN mode).
        """

        try:
            if use_dsn:
                engine = connect_server(dsn_name=dsn_header_str)
            else:
                engine = connect_server(driver="FreeTDS",
                                        server=sql_server_str,
                                        port=port,
                                        tds_version=7.1,
                                        database=sql_database_str)
            engine = ensure_connection_is_engine(engine)
            self.db_engine = engine

        ### TODO: This is not actually catching if the engine exists or not...
        ### add raise exception to rtb-db login.connect_sqlalchemy
        except Exception as err:
            logging.warning('Unable to connect to RTB database.')
            logging.error(f"Received {err}")
            raise ConnectionError(
                f"Unable to connect to RTB database - {err}."
            )

    def new_pipeline_db_entry(self, ref_type, wfi_mode, reef_monitor):
        """
        Initialize and insert a new logistics processing table row for this pipeline.

        Parameters
        ----------
        ref_type : str
            Reference type associated with this pipeline run.
        wfi_mode : str
            wfi mode associated with this pipeline run.
        reef_monitor : bool
            Expecting external monitoring for this run.
        """

        self.db_entry.init_rfp_log_pro(ref_type, wfi_mode, reef_monitor)
        add_to_tables_from_class_list(self.db_engine, [self.db_entry.rfp_log_pro])