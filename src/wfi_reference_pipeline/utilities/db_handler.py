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
    Utility class to safely handle access to the rtbdb
    This class is designed to ONLY be initialized in the Pipeline base class
    This class should be initialized as part of a base pipeline initialization procedure AFTER config values have been read and stored
    """

    def __init__(self, ref_type, use_dsn, sql_server_str=None, sql_database_str=None, port=None, dsn_header_str=None):
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
        """Confirm if the user has access to the desired database. If so, establish a SQLalchemy connection.
        Saves SQLalchemy engine to self.db_engine and saves the db availability boolean to self.use_db.

        Returns
        -------
        self.use_db
            boolean to trigger if montor should attempt to use the database

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

            # TODO - SAPP -> TEMP DEV CODE REMOVE
            print(f'SQL table names: {table_tools.table_names(engine)}\n')
            print(f'Table class names: {table_tools.table_class_names(engine)}\n')
            print(f'Table name from class: {RFPLogProTable.__tablename__}')
            print(f"Table class from name: {table_tools.get_table_class_from_tablename('wfi_rfp_log_pro')}")
            print(table_tools.get_full_table(engine, RFPLogProTable))
            #//////////////////////////////////////////////////////////////////



        ### TODO: This is not actually catching if the engine exists or not...
        ### add raise exception to rtb-db login.connect_sqlalchemy
        except Exception as err:
            logging.warning('Unable to connect to RTB database.')
            logging.error(f"Received {err}")
            raise ConnectionError(
                f"Unable to connect to RTB database - {err}."
            )

    def new_pipeline_db_entry(self, ref_type, wfi_mode, reef_monitor):
        # SAPP TODO - YOU ARE HERE!!!! GET AN INITIAL DATABASE ENTRY AND KEEP THE SQL ID
        # STORE EVERYTHING YOU HAVE ALREADY PERTAINING TO THIS PIPELINE RUN
        # CREATE A METHOD IN RTB_DB REPO THAT WILL ALLOW YOU TO UPDATE AN EXISTING ROW
        # ONCE YOU HAVE INITIAL UPDATE AND EXISTING ROW UPDATE, TRY RUNNING AND POPULATING EVERYTHING NEEDED FROM INITIATION THROUGH PREP PIPELINE

        self.db_entry.init_rfp_log_pro(ref_type, wfi_mode, reef_monitor)
        add_to_tables_from_class_list(self.engine, [self.db_entry.rfp_log_pro])
        #self.sql_id = self.rfp_log_pro.sql_id   # VERIFY SQL_ID EXISTS
