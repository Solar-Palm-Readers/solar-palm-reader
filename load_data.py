import os
from datetime import datetime

import numpy as np
import pandas as pd
from spacepy import pycdf

data_root = r'data'
dscovr_file_template = 'dscovr_h0_mag_{date_string}_v01.cdf'
wind_mfi_file_template = 'wi_h2_mfi_{date_string}_v04.cdf'
wind_swe_file_template = 'wi_h1_swe_{date_string}_v01.cdf'


def date_to_string(date: datetime) -> str:
    return date.strftime('%Y%m%d')


def get_file(dataset: str, date: datetime) -> pycdf.CDF:
    if dataset.lower() == "dscovr":
        file_template = dscovr_file_template
    elif dataset.lower() == "wind mfi":
        file_template = wind_mfi_file_template
    elif dataset.lower() == "wind swe":
        file_template = wind_swe_file_template
    else:
        raise ValueError('Unknown Dataset')

    file_name = str.format(file_template, date_string=date_to_string(date))
    full_file_path = os.path.join(data_root, dataset.lower(), file_name)
    return pycdf.CDF(full_file_path)


def get_file_dscovr(date: datetime) -> pycdf.CDF:
    return get_file("dscovr", date)


def get_file_wind_mfi(date: datetime) -> pycdf.CDF:
    return get_file("wind mfi", date)


def get_file_wind_swe(date: datetime) -> pycdf.CDF:
    return get_file("wind swe", date)


def display_columns(dat_files: list, metadata: bool = False) -> None:
    for f in dat_files:
        print(f.attrs['Descriptor'])
        for i in f.keys():
            try:
                info = f[i].attrs['CATDESC']
            except:
                info = f[i].attrs['FIELDNAM']
            if not metadata:
                if f[i].attrs['VAR_TYPE'] == 'metadata':
                    continue
            print(i, info, f[i].shape)


def to_pandas(dat: pycdf.CDF) -> pd.DataFrame:
    df = pd.DataFrame()
    for k in dat.keys():
        col = dat[k]
        if col.attrs['VAR_TYPE'] == 'metadata':
            continue
        if col[:].ndim > 1:
            for c in range(col.shape[-1]):
                df[f'{k}_{c}'] = col[:, c]
        else:
            df[k] = col[:].tolist()

    return df


def load_dataframe(dataset: str, date: datetime) -> pd.DataFrame:
    dat_file = get_file(dataset, date)
    df = to_pandas(dat_file)
    dat_file.close()
    return df


def validate_data(dat):  # not finished
    validate_vector = np.zeros_like(dat['B1F1'])
    for k in dat.keys():
        col = dat[k]
        if col.attrs['VAR_TYPE'] == 'metadata':
            continue
        fill_val = col.attrs['FILLVAL']
        max_val = col.attrs['VALIDMAX']
        min_val = col.attrs['VALIDMIN']
        if col[:].ndim > 1:
            valid = np.logical_or(
                np.sum(col[:] > max_val, axis=1), np.sum(col[:] == fill_val, axis=1))
            valid = np.logical_or(valid, np.sum(col[:] < min_val, axis=1))
            validate_vector = np.logical_or(valid, validate_vector)
        else:
            valid = np.logical_or(col[:] > max_val, col[:] == fill_val)
            valid = np.logical_or(valid, col[:] < min_val)
            validate_vector = np.logical_or(valid, validate_vector)

        return np.sum(validate_vector)


if __name__ == '__main__':
    # wind_mfi_file = get_file_wind_mfi(datetime(year=2022, month=1, day=1))
    # wind_swe_file = get_file_wind_swe(datetime(year=2022, month=1, day=1))

    # display_columns([dscovr_file, wind_mfi_file, wind_swe_file], metadata=True)
    df = load_dataframe(datetime(year=2022, month=1, day=1))
    print('hi')
