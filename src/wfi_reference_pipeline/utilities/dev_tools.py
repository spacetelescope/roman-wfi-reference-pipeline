import logging
import os

import psutil

from wfi_reference_pipeline.constants import GB


def get_mem_usage_gb():
    """
    Function to return memory usage throughout module.

    Returns
    ----------
    memory_usage; float
        Memory in Gigabytes being used.

    """
    memory_usage = psutil.virtual_memory().used / GB  # in GB
    return memory_usage

def get_process_memory_usage(self):
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    rss = memory_info.rss  # Resident Set Size: physical memory used
    vms = memory_info.vms  # Virtual Memory Size: total memory used, including swap
    return rss, vms

def log_process_memory_usage(self):
    rss, vms = self.get_process_memory_usage()
    logging.debug(
        f"Memory usage: RSS={rss / (1024 ** 2):.2f} MB, VMS={vms / (1024 ** 2):.2f} MB"
    )