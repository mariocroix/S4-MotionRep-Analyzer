import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/02_outliers_removed_chauvenets.pkl")

predictor_columns = list(df.columns[:6])

# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------
# basically adds a linear line between the gaps in data plots 

for col in predictor_columns:
    df[col] = df[col].interpolate()
    
df.info()

# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------

df[df["set"] == 25]["acc_y"].plot()

duration = df[df["set"] == 1].index[-1] - df[df["set"] == 1].index[0]
duration.seconds

for s in df["set"].unique():
    
    start = df[df["set"] == s].index[0]
    stop = df[df["set"] == s].index[-1]
    
    duration = stop - start
    df.loc[(df["set"] == s), "duration"] = duration.seconds
    
duration_df = df.groupby(["category"])["duration"].mean()

duration_df.iloc[0] / 5
duration_df.iloc[1] / 10

# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------

df.lowpass = df.copy()
LowPass = LowPassFilter()

# 5 instances per sec
fs = 1000 / 200
# by changing this the we change how much data should be filtered (the less the number the more data filtered)

cutoff = 1.2

df_lowpass = LowPass.low_pass_filter(df.lowpass, "acc_y", fs, cutoff, order=5)

#  see the difference between filtered and not filtered data

subset = df_lowpass[df_lowpass["set"] == 45]
print(subset["label"][0])

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
ax[0].plot(subset["acc_y"].reset_index(drop=True), label="raw data")
ax[1].plot(subset["acc_y_lowpass"].reset_index(drop=True), label="butterworth filter")
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)

# override all the cols with filtered data
for col in predictor_columns:
    df_lowpass = LowPass.low_pass_filter(df_lowpass, col, fs, cutoff, order=5)
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    del df_lowpass[col + "_lowpass"]

# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------

df_pca = df_lowpass.copy()
PCA = PrincipalComponentAnalysis()

# determine the optimal number of principal components
# the goal is to reduce the num of cols

pc_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)
# the elbow technique, a perfect example of how it works
# 3 is the optimal num for the principal components
plt.figure(figsize=(10, 10))
plt.plot(range(1, len(predictor_columns) + 1), pc_values)
plt.xlabel("principal component number")
plt.ylabel("explained variance")
plt.show()

# keep this here, when we'll do feature selection, we'll check up whether
# principal components perform better than the indiv. cols
df_pca = PCA.apply_pca(df_pca, predictor_columns, 3)

subset = df_pca[df_pca["set"] == 35]
subset[["pca_1", "pca_2", "pca_3"]].plot()

# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------

# it is impartial to orientation
df_squared = df_pca.copy()

acc_r = df_squared["acc_x"] ** 2 + df_squared["acc_y"] ** 2 + df_squared["acc_z"] ** 2 
gyr_r = df_squared["gyr_x"] ** 2 + df_squared["gyr_y"] ** 2 + df_squared["gyr_z"] ** 2 

df_squared["acc_r"] = np.sqrt(acc_r)
df_squared["gyr_r"] = np.sqrt(gyr_r)

subset = df_squared[df_squared["set"] == 14]
subset[["acc_r", "gyr_r"]].plot(subplots=True)

del df_squared["duration"]

# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------

df_temporal = df_squared.copy()
NumAbs = NumericalAbstraction()

predictor_columns = predictor_columns + ["acc_r", "gyr_r"]

# using a large num for window size is missing a lot of data
# window size is how many values we want to look back, the mean? will be calc-ed based on that values
ws = int(1000 / 200)

# this one is mixing different set(exercice) data so we gotta put into subset chunks(for each exercice) then concat back
for col in predictor_columns:
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "mean" )
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "std" )

df_temporal_list = []
for s in df_temporal["set"].unique():
    subset = df_temporal[df_temporal["set"] == s].copy()
    for col in predictor_columns:
        subset = NumAbs.abstract_numerical(subset, [col], ws, "mean" )
        subset = NumAbs.abstract_numerical(subset, [col], ws, "std" )
    df_temporal_list.append(subset)

df_temporal = pd.concat(df_temporal_list)

# since we used mean and std, the data looks smoother
subset[["acc_y", "acc_y_temp_mean_ws_5", "acc_y", "acc_y_temp_std_ws_5"]].plot()
subset[["gyr_y", "gyr_y_temp_mean_ws_5", "gyr_y", "gyr_y_temp_std_ws_5"]].plot()

# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------

df_freq = df_temporal.copy().reset_index()
FreqAbs = FourierTransformation()

# window size and sampling rate
fs = int(1000 / 200)
ws = int(2800 / 200) 

df_freq = FreqAbs.abstract_frequency(df_freq, ["acc_y"], ws, fs)

df_freq_list = []
for s in df_freq["set"].unique():
    print(f"fourier {s}")
    subset = df_freq[df_freq["set"] == s].reset_index(drop=True).copy()
    subset = FreqAbs.abstract_frequency(subset, predictor_columns, ws, fs)
    df_freq_list.append(subset)

df_freq = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)
# this results in highly corellated data in rows

# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------

df_freq = df_freq.dropna()
#removes every other row to reduce the corel. between the adjacent records
df_freq = df_freq.iloc[::2]

# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------

df_cluster = df_freq.copy()

cluster_columns = ["acc_x", "acc_y", "acc_z"]
k_values = range(2, 10)
inertias = []

for k in k_values:
    subset = df_cluster[cluster_columns]
    # train the model
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    cluster_labels = kmeans.fit_predict(subset)
    # export inertias
    inertias.append(kmeans.inertia_)
    
# elbow methos based on inertias
plt.figure(figsize=(10, 10))
plt.plot(k_values, inertias)
plt.xlabel("k")
plt.ylabel("Sum of squared distances")
plt.show()

kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)
subset = df_cluster[cluster_columns]
df_cluster["cluster"] = kmeans.fit_predict(subset)
    
# Plot clusters
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for c  in df_cluster["cluster"].unique():
    subset = df_cluster[df_cluster["cluster"] == c]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=c)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()

# Plot accelerometer data to compare
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for c  in df_cluster["label"].unique():
    subset = df_cluster[df_cluster["label"] == c]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=c)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()
# we can see here that bench and ohp movements are quite similar
# dead and row sim too
# squat is completely dif
# rest is unexpected so it is scattered
 
# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

df_cluster.to_pickle("../../data/interim/03_data_features.pkl")