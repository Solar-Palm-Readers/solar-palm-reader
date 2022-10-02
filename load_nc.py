import glob
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr
from cdflib import cdf_to_xarray
from scipy.io import netcdf

data_root = r'data'
dscovr_fc0_file = 'oe_fc0_dscovr_s{date_string}000000_*_pub.nc'
dscovr_fc1_file = 'oe_fc1_dscovr_s{date_string}000000_*_pub.nc'


def date_to_string(date: datetime) -> str:
    return date.strftime('%Y%m%d')


def get_file(dataset: str, date: datetime) -> xr.Dataset:
    if dataset.lower() == "dscovr fc0":
        file_template = dscovr_fc0_file
        with_wildcard = True
    elif dataset.lower() == "dscovr fc1":
        file_template = dscovr_fc1_file
        with_wildcard = True
    else:
        raise ValueError('Unknown Dataset')

    file_name = str.format(file_template, date_string=date_to_string(date))
    full_file_path = os.path.join(data_root, dataset.lower(), file_name)

    if not with_wildcard:
        return cdf_to_xarray(full_file_path, to_datetime=True, fillval_to_nan=True)
    else:
        full_file_path = glob.glob(full_file_path)[0]
        return netcdf.NetCDFFile(full_file_path, 'r')


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


def to_pandas(dat: xr.Dataset, validate_vector: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame()
    for k in dat.variables.keys():
        col = dat.variables[k]
        df[k] = col[:]
    df['validate_vector'] = ~validate_vector
    return df


def load_dataframe(dataset: str, date: datetime) -> pd.DataFrame:
    dat_file = get_file(dataset, date)
    validate_vector = get_validate_vector(dat_file)
    df = to_pandas(dat_file, validate_vector)
    dat_file.close()
    return df


def get_validate_vector(dat: xr.Dataset) -> np.array:
    validate_vector = np.zeros_like(dat.variables['time'][:])
    for k in dat.variables.keys():
        col = dat.variables[k]
        try:
            max_val = col.valid_max
            min_val = col.valid_min
        except:
            continue
        valid = np.logical_or(col[:] > max_val, col[:] < min_val)
        validate_vector = np.logical_or(valid, validate_vector)

    return validate_vector


def load_multiple_files(dataset: str, start_date: datetime, stop_date: datetime) -> pd.DataFrame:
    date = start_date
    df = pd.DataFrame()
    while date < stop_date:
        new_df = load_dataframe(dataset, date)
        df = pd.concat([df, new_df])
        date = date + timedelta(days=1)
    return df


if __name__ == '__main__':
    # wind_mfi_file = get_file_wind_mfi(datetime(year=2022, month=1, day=1))
    # wind_swe_file = get_file_wind_swe(datetime(year=2022, month=1, day=1))

    # display_columns([dscovr_file, wind_mfi_file, wind_swe_file], metadata=True)
    df1 = load_multiple_files('dscovr fc0', datetime(year=2022, month=1, day=1), datetime(year=2022, month=1, day=3))
    print('hi')
