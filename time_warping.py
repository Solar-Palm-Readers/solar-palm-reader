from typing import Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('TkAgg')
import dtw

from load_data import load_saved_dataframe


def get_time_vector(dscovr: pd.DataFrame, wind_mfi: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate an appropriate pairing of the DSCOVR and Wind times based on the magnetic feild data.
    """
    resampled_dscovr = dscovr.resample('120s', on='Epoch1').mean()
    resampled_wind_mfi = wind_mfi.resample('120s', on='Epoch').mean()

    batch_size = 60 * 12 // 2  # Half a day
    num_batches = resampled_dscovr.shape[0] // batch_size

    dscovr_time_vectors = []
    warped_wind_time_vectors = []

    for i in range(num_batches):
        print(f'{i}/{num_batches}')
        dscovr_batch = resampled_dscovr['B1GSE_2'].iloc[i * batch_size:(i + 1) * batch_size]
        wind_batch = resampled_wind_mfi['BGSE_2'].iloc[i * batch_size:(i + 1) * batch_size]

        dscovr_batch.fillna(value=dscovr_batch.median())
        wind_batch.fillna(value=wind_batch.median())

        alignment = dtw.dtw(dscovr_batch, wind_batch, keep_internals=True, step_pattern='symmetricP05')

        # alignment.plot(type="twoway", offset=-10)
        # plt.show()
        # alignment.plot(type="threeway")
        # plt.show()

        dscovr_time = dscovr_batch.index.to_numpy()[:-1]
        dscovr_time_vectors.append(dscovr_time)

        wq = dtw.warp(alignment, index_reference=True)
        warped_wind_time = wind_batch.iloc[wq].index.to_numpy()
        warped_wind_time_vectors.append(warped_wind_time)

    dscovr_time = np.concatenate(dscovr_time_vectors).ravel()
    warped_wind_time = np.concatenate(warped_wind_time_vectors).ravel()

    return dscovr_time, warped_wind_time


def warp_target_values(wind_swe: pd.DataFrame, dscovr_time: np.ndarray, warped_wind_time: np.ndarray) -> pd.DataFrame:
    """
    Warp the epoch times from the Wind SWE dataset to the matching times from DSCOVR.
    """
    wind_swe = wind_swe.copy()
    # Remove occurrences outside the scope of the time warping
    wind_swe = wind_swe[np.bitwise_and(wind_swe['Epoch'] > np.min(warped_wind_time),
                                       wind_swe['Epoch'] < np.max(warped_wind_time))]

    actual_wind_times = wind_swe['Epoch'].to_numpy()
    indices = np.searchsorted(warped_wind_time, actual_wind_times)

    # Calculate the fraction of each time in the segment containing it
    fractions_in_segments = ((actual_wind_times - warped_wind_time[indices - 1])
                             / (warped_wind_time[indices] - warped_wind_time[indices - 1]))

    # Use linear interpolation to get a more accurate warping of the time
    warped_wind_times = (dscovr_time[indices - 1]
                         + fractions_in_segments * (dscovr_time[indices] - dscovr_time[indices - 1]))

    wind_swe['Epoch_old'] = actual_wind_times
    wind_swe['Epoch'] = warped_wind_times
    return wind_swe


if __name__ == '__main__':
    df_dscovr = load_saved_dataframe('dscovr example')
    df_wind_mfi = load_saved_dataframe('wind mfi example')
    df_wind_swe = load_saved_dataframe('wind swe example')

    df_dscovr_time, df_warped_wind_time = get_time_vector(df_dscovr, df_wind_mfi)
    df_warped_wind_swe = warp_target_values(df_wind_swe, df_dscovr_time, df_warped_wind_time)

    print('end')
