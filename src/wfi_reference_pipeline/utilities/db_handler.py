import logging

from rtb_db.utilities import table_tools
from rtb_db.utilities.login import connect_server
from rtb_db.utilities.table_tools import ensure_connection_is_engine

from wfi_reference_pipeline import constants


class DBHandler:
    """
    Utility class to safely handle access to the rtbdb
    This class is designed to ONLY be initialized in the Pipeline base class
    This class should be initialized as part of a base pipeline initialization procedure AFTER config values have been read and stored
    """

    def __init__(self, ref_type, sql_server_str, sql_database_str, dsn_header_str):
        if ref_type not in constants.WFI_REF_TYPES:
            raise ValueError(
                f"ref_type {ref_type} not valid, must be: {constants.WFI_REF_TYPES}"
            )

        self.ref_type = ref_type
        self.sql_engine = None
        self.sql_server_str = sql_server_str
        self.sql_database_str = sql_database_str
        self.dsn_header_str = dsn_header_str
        self.connect()

    def connect(self):
        """Confirm if the user has access to the desired database. If so, establish a SQLalchemy connection.
        Saves SQLalchemy engine to self.sql_engine and saves the db availability boolean to self.use_db.

        Returns
        -------
        self.use_db
            boolean to trigger if montor should attempt to use the database

        """
        try:
            if self.dsn_header_str:
                engine = connect_server(dsn_name=self.dsn_header_str)
            else:
                engine = connect_server(driver="FreeTDS",
                                        server=self.sql_server_str,
                                        port=1433,
                                        tds_version=7.1,
                                        database=self.sql_database_str)
            engine = ensure_connection_is_engine(engine)
            self.sql_engine = engine

            # TODO - SAPP -> TEMP DEV CODE REMOVE
            print(f'SQL table names: {table_tools.table_names(engine)}\n')
            print(f'Table class names: {table_tools.table_class_names(engine)}\n')
            #//////////////////////////////////////////////////////////////////

            # ensure metrics table exists on the active database server
            # assert set(self.sql_table_names).issubset(table_names(self.sql_engine)) # TODO figure this out, not currently storing sql_table_names


        ### TODO: This is not actually catching if the engine exists or not...
        ### add raise exception to rtb-db login.connect_sqlalchemy
        except Exception as err:
            logging.warning('Unable to connect to RTB database.')
            logging.error(f"Received {err}")
            print(f"Received {err}")
            ## TODO - IF YOU DO GET HERE, do you want to abandon ship and alert?
            return

