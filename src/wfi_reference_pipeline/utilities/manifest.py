import pandas as pd
from roman_datamodels import datamodels as rdd


def make_manifest(files):
    """
    Function for returning a pandas dataframe containing important
    metadata inside Roman reference files.

    Parameters
    ----------
    files: list of strings
        List of reference file names from which to get metadata.

    Returns
    -------
    df: pandas.DataFrame
        Pandas dataframe containing the file names, reftypes, detector names,
        exposure types, optical elements, MA table names, and
        useafter dates for the input file list.
    """

    useafter, exptype, element, detector = [], [], [], []
    ma_name, reftype = [], []

    for file in files:
        with rdd.open(file) as rf:
            meta = rf.meta

        reftype.append(meta['reftype'])
        useafter.append(meta['useafter'].datetime.strftime('%Y-%m-%d %H:%M:%S'))
        detector.append(meta['instrument']['detector'])
        try:
            exptype.append(meta['exposure']['type'])
        except KeyError:
            exptype.append('N/A')
        try:
            element.append(meta['instrument']['optical_element'])
        except KeyError:
            element.append('N/A')
        try:
            ma_name.append(meta['observation']['ma_table_name'])
        except KeyError:
            ma_name.append('N/A')

    df = pd.DataFrame({'file': files, 'reftype': reftype, 'detector': detector,
                       'exptype': exptype, 'optical_element': element,
                       'ma_table_name': ma_name, 'useafter': useafter})

    return df
