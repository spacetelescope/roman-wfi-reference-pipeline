import datetime
import getpass
import logging
import os
import pwd
import socket
import sys
import time

from functools import wraps


def configure_logging(target_module, path=None, level=logging.INFO):
    """Configure the standard logging format.

    Parameters
    ----------
    target_module (string):
        The name of the module being logged.
    path (string):
        Where to write the log if user-supplied path; default to working dir.
    level (integer):
        Minimum logging level to display messages. These are technically
        integers, but can use inputs like `logging.INFO` or `logging.DEBUG`.

    Returns
    -------
    log_file (string):
        The name and path of the log file.
    """

    # Added this to make sure nothing is getting in the way of our log file.
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)

    log_file = make_log_file(target_module, path=path)

    logging.basicConfig(filename=log_file,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S %p',
                        level=level,
                        filemode='a')

    logging.captureWarnings(False)

    return log_file


def make_log_file(target_module, path=None):
    """Return the name of the log file based on the module name.

    The name of the ``log_file`` is a combination of the name of the
    module being logged, the user running the code, and the current datetime.

    Parameters
    ----------
    target_module : str
        The name of the module being logged.
    path : str
        Where to write the log if user-supplied path; default to working dir.

    Returns
    -------
    log_file : str
        The full path to where the log file will be written to.
    """

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    user = pwd.getpwuid(os.getuid()).pw_name
    filename = f'{target_module}_{user}_{timestamp}.log'

    if not path:
        log_file = os.path.join(os.getcwd(), filename)
    else:
        log_file = os.path.join(path, filename)

    return log_file


def log_info(func):
    """Decorator to log potentially useful system information.

    This function can be used as a decorator to log user environment
    and system information.

    Parameters
    ----------
    func : func
        The function to decorate.

    Returns
    -------
    wrapped : func
        The wrapped function.
    """
    @wraps(func)
    def wrapped(*a, **kw):

        # Log environment information
        logging.info('User: ' + getpass.getuser())
        logging.info('System: ' + socket.gethostname())
        logging.info('Python Version: ' + sys.version.replace('\n', ''))
        logging.info('Python Executable Path: ' + sys.executable)

        # Call the function and time it
        t1_cpu = time.process_time()
        t1_time = time.time()
        func(*a, **kw)
        t2_cpu = time.process_time()
        t2_time = time.time()

        # Log execution time
        hours_cpu, remainder_cpu = divmod(t2_cpu - t1_cpu, 60 * 60)
        minutes_cpu, seconds_cpu = divmod(remainder_cpu, 60)
        hours_time, remainder_time = divmod(t2_time - t1_time, 60 * 60)
        minutes_time, seconds_time = divmod(remainder_time, 60)
        logging.info(f'Elapsed Real Time: {hours_time:.0f}:'
                     f'{minutes_time:.0f}:{seconds_time:f}')
        logging.info(f'Elapsed CPU Time: {hours_cpu:.0f}:'
                     f'{minutes_cpu:.0f}:{seconds_cpu:f}')

    return wrapped
