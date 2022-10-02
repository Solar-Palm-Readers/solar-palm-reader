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
    with_wildcard = False
    if dataset.lower() == "dscovr fc0":
        file_template = dscovr_fc0_file
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


def to_pandas(dat: xr.Dataset, columns: list, validate_vector: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame()
    for col_name in columns:
        col = dat[col_name]
        if col.data.ndim > 1:
            for c in range(col.shape[-1]):
                df[f'{col_name}_{c}'] = col.data[:, c]
        else:
            df[col_name] = col.data
    df['validate_vector'] = ~validate_vector
    return df


def add_epoch(dataset: str, df: pd.DataFrame, dat: xr.Dataset) -> None:
    if dataset.lower() == "dscovr":
        df['Epoch1'] = dat.coords['Epoch1'].data
    elif dataset.lower() == "wind mfi":
        df['Epoch'] = dat.coords['Epoch'].data
    elif dataset.lower() == "wind swe":
        df['Epoch'] = dat.coords['Epoch'].data
    else:
        raise ValueError('Unknown Dataset')


def load_dataframe(dataset: str, date: datetime) -> pd.DataFrame:
    dat_file = get_file(dataset, date)
    columns = select_columns(dataset)
    validate_vector = get_validate_vector(dataset, dat_file, columns)
    df = to_pandas(dat_file, columns, validate_vector)
    add_epoch(dataset, df, dat_file)
    dat_file.close()
    return df


def select_columns(dataset: str) -> list:
    if dataset.lower() == "dscovr":
        columns = ['B1F1', 'B1SDF1', 'B1GSE', 'B1SDGSE']
    elif dataset.lower() == "wind mfi":
        columns = ['BF1', 'BGSE']
    elif dataset.lower() == "wind swe":
        columns = ['Proton_VX_nonlin', 'Proton_VY_nonlin', 'Proton_VZ_nonlin', 'Proton_W_nonlin', 'Proton_Np_nonlin']
    else:
        raise ValueError('Unknown Dataset')

    return columns


def create_validate_vector(dataset: str, dat: xr.Dataset) -> np.ndarray:
    if dataset.lower() == "dscovr":
        validate_vector = np.zeros_like(dat['B1F1'])
    elif dataset.lower() == "wind mfi":
        validate_vector = np.zeros_like(dat['BF1'])
    elif dataset.lower() == "wind swe":
        validate_vector = np.zeros_like(dat['Proton_VX_nonlin'])
    else:
        raise ValueError('Unknown Dataset')

    return validate_vector


def get_validate_vector(dataset: str, dat: xr.Dataset, columns: list) -> np.array:
    validate_vector = create_validate_vector(dataset, dat)
    for col_name in columns:
        col = dat[col_name]
        fill_val = col.attrs['FILLVAL']
        max_val = col.attrs['VALIDMAX']
        min_val = col.attrs['VALIDMIN']
        if col.ndim > 1:
            valid = np.logical_or(
                np.sum(col.data > max_val, axis=1), np.sum(col.data == fill_val, axis=1))
            valid = np.logical_or(valid, np.sum(col.data < min_val, axis=1))
        else:
            valid = np.logical_or(col.data > max_val, col.data == fill_val)
            valid = np.logical_or(valid, col.data < min_val)
        validate_vector = np.logical_or(valid, validate_vector)

    return validate_vector


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df['validate_vector']]
    df = df.drop('validate_vector', axis=1)

    return df


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
