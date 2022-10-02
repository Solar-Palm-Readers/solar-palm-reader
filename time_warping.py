from datetime import datetime

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from dtw import *

from load_cdf import load_dataframe

# from main import get_file_dscovr,get_file_wind_mfi

dsc = load_dataframe('dscovr', datetime(year=2022, month=1, day=1))
wind_mfi = load_dataframe('wind mfi', datetime(year=2022, month=1, day=1))
wind_swe = load_dataframe('wind swe', datetime(year=2022, month=1, day=1))

vec = []
dsc4 = dsc.resample('60S', on='Epoch1').mean()
wind4 = wind_mfi.resample('60S', on='Epoch').mean()

batch_size = 720
num_batches = dsc4.shape[0] // batch_size

for i in range(num_batches):
    dscovr_batch = dsc4['B1GSE_0'].iloc[i * batch_size:(i + 1) * batch_size]
    wind_batch = wind4['BGSE_0'].iloc[i * batch_size:(i + 1) * batch_size]
    alignment = dtw(dscovr_batch, wind_batch, keep_internals=True, step_pattern='symmetricP05')

    alignment.plot(type="twoway", offset=-10)
    plt.show()
    alignment.plot()
    plt.show()
    w = warp(alignment, index_reference=True)

    wind_warp = wind_mfi.iloc[w]
    wind_warp = wind_warp.reset_index()
    """
    for j in range(wind_warp.shape[0]):
        x = wind_warp['Epoch']
        difference = np.abs(wind_swe['Epoch'] - x)
        index = difference.argmin()
        vec[j] = index
    """
# x = dsc4['B1GSE_0'].iloc[:599]
# x = x.reset_index()
# wind_warp = wind_warp.reset_index()
# coef = np.corrcoef(x['B1GSE_0'].to_numpy(),wind_warp['BGSE_0'].to_numpy())
v = 3
