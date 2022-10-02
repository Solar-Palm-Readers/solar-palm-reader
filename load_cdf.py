import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr
from cdflib import cdf_to_xarray

data_root = r'data'
dscovr_file_template = 'dscovr_h0_mag_{date_string}_v01.cdf'
wind_mfi_file_template = 'wi_h2_mfi_{date_string}_v04.cdf'
wind_swe_file_template = 'wi_h1_swe_{date_string}_v01.cdf'


def date_to_string(date: datetime) -> str:
    return date.strftime('%Y%m%d')


def get_file(dataset: str, date: datetime) -> xr.Dataset:
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
    return cdf_to_xarray(full_file_path, to_datetime=True, fillval_to_nan=False)


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
    return filter_data(df)


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
            valid = np.logical_or(valid, np.sum(col.data == np.nan, axis=1))
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
        print(date)
        new_df = load_dataframe(dataset, date)
        df = pd.concat([df, new_df])
        date = date + timedelta(days=1)
    return df


if __name__ == '__main__':
    # display_columns([dscovr_file, wind_mfi_file, wind_swe_file], metadata=True)
    df = load_dataframe('dscovr', datetime(year=2022, month=1, day=1))

    month = 1
    name = 'jan'

    df = load_multiple_files('dscovr', datetime(year=2022, month=month, day=1),
                             datetime(year=2022, month=month, day=4))
    df.to_pickle(os.path.join(data_root, 'dscovr_example.dat'))

    df = load_multiple_files('wind swe', datetime(year=2022, month=month, day=1),
                             datetime(year=2022, month=month, day=4))
    df.to_pickle(os.path.join(data_root, 'wind_swe_example.dat'))

    df = load_multiple_files('wind mfi', datetime(year=2022, month=month, day=1),
                             datetime(year=2022, month=month, day=4))
    df.to_pickle(os.path.join(data_root, 'wind_mfi_example.dat'))

    # df = load_multiple_files('dscovr', datetime(year=2022, month=month, day=1),
    #                          datetime(year=2022, month=month + 1, day=1))
    # df.to_pickle(os.path.join(data_root, f'dscovr_{name}.dat'))

    # df = load_multiple_files('wind swe', datetime(year=2022, month=month, day=1),
    #                          datetime(year=2022, month=month + 1, day=1))
    # df.to_pickle(os.path.join(data_root, f'wind_swe_{name}.dat'))

    # df = load_multiple_files('wind mfi', datetime(year=2022, month=month, day=1),
    #                          datetime(year=2022, month=month + 1, day=1))
    # df.to_pickle(os.path.join(data_root, f'wind_mfi_{name}.dat'))
