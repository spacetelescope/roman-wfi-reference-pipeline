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
    ma_name, reftype, description = [], [], []

    for file in files:
        with rdd.open(file) as rf:
            meta = rf.meta

        reftype.append(meta['reftype'])
        useafter.append(meta['useafter'].datetime.strftime('%Y-%m-%d %H:%M:%S'))
        detector.append(meta['instrument']['detector'])
        description.append(meta['description'])
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

    df = pd.DataFrame({'file': files,
                       'reftype': reftype,
                       'detector': detector,
                       'descirption': description,
                       'useafter': useafter,
                       'exptype': exptype,
                       'optical_element': element,
                       'ma_table_name': ma_name}
                      )
    print(df)
    return df


def print_manifest(df):
    """
    Prints each file's meta data separated by a line.
    """
    for index, row in df.iterrows():
        print(f"File: {row['file']}")
        print(f"Reftype: {row['reftype']}")
        print(f"Detector: {row['detector']}")
        print(f"Description: {row['descirption']}")
        print(f"Use After: {row['useafter']}")
        print(f"Exptype: {row['exptype']}")
        print(f"Optical Element: {row['optical_element']}")
        print(f"MA Table Name: {row['ma_table_name']}")
        print('-' * 40)  # Separator line between rows


def print_meta_fields_together(df):
    """
    Prints the meta of every file by key in groups and separates them by a line.
    """
    for col in df.columns:
        print(f"{col.capitalize()}:")
        for value in df[col]:
            print(f"  {value}")
        print('-' * 40)   # Separator line between columns
