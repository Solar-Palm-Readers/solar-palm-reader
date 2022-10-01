import os
from datetime import datetime

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


if __name__ == '__main__':
    cdf_file = get_file_dscovr(datetime(year=2022, month=1, day=1))

    print(cdf_file)
