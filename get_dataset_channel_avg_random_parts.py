import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import random
import warnings
import os
warnings.filterwarnings('ignore')

np.set_printoptions(formatter={'float_kind':'{:f}'.format})

fs = 256
n = 64
k = 0
iterations = 5
outfile = "dataset_train_" + str(n) + ".csv"
outfile = "dataset_test_1_2.csv"
trainfiles = ["0_1","0_2","0_3","1_1","1_2","1_3"]


files = trainfiles
files = ["1_test_2"]
songids = {"0_1": 0, "0_2": 0, "0_test": 0, "1_1": 1, "1_2": 1, "1_test": 1, "0_3": 0, "1_3": 1, "0_test_2": 0, "1_test_2": 1}
channels = ["TP9", "AF7", "AF8", "TP10"]
while True:
    for fname in files:
        filename = "recordings/" + fname
        raw_data = pd.read_csv(filename+".csv")

        for m in range(iterations):
            values = dict()
            values["songid"] = songids[fname]
            l = 0

            start = round((random.randint(10,65)/100)*len(raw_data))
            end = start + random.randint(start, len(raw_data))
            if end > len(raw_data) or end - start < 5000:
                end = len(raw_data)

            raw_data = pd.read_csv(filename+".csv").loc[start:end].reset_index()
            del raw_data["timestamps"]
            del raw_data["Right AUX"]

            data = raw_data.loc[:, channels[0]]
            eeg_bands = {'Delta': (0, 4),
                         'Theta': (4, 8),
                         'Alpha': (8, 12),
                         'Beta': (12, 30),
                         'Gamma': (30, 45)}

            eeg_band_fft = dict()
            avgs = dict()
            channelavgs = []
            for band in eeg_bands:
                eeg_band_fft[band] = dict()
                for channel in channels:
                    eeg_band_fft[band][channel] = []

            for channel in channels:
                avgs[channel] = []

            for i in range(n, len(data), n):
                for channel in channels:
                    locdata = raw_data.loc[i-n:i, channel]
                    fft_vals = np.absolute(np.fft.rfft(locdata))
                    fft_freq = np.fft.rfftfreq(len(locdata), 1.0/fs)
                    for band in eeg_bands:
                        freq_ix = np.where((fft_freq >= eeg_bands[band][0]) &
                                           (fft_freq <= eeg_bands[band][1]))[0]
                        eeg_band_fft[band][channel].append(np.mean(fft_vals[freq_ix]))

            for band in eeg_bands:
                for channel in channels:
                    avgs[channel].append(np.average(np.array(eeg_band_fft[band][channel])))

            del eeg_band_fft["Delta"]

            for channel in channels:
                avg = np.round(preprocessing.MinMaxScaler().fit_transform(np.array(avgs[channel]).reshape(-1,1)), decimals=4)[1:-1]
                channelavgs.append(avg)
                l = 0
                for average in avg:
                    values[channel+str(l)] = average
                    l+=1

            if not np.isnan(np.sum(np.array(channelavgs))):
                k+=1
                dataframe = pd.DataFrame(values, index = [0])
                dataframe.to_csv(outfile, index=False, mode='a', header=not os.path.isfile(outfile))
