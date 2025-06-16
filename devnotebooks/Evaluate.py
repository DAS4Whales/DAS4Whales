# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: venv
#     language: python
#     name: python3
# ---

import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
plt.rcParams['font.size'] = 24

# +
with open('../denoised_data/Batch_1/association_2021-11-04_02:00:02.pkl', 'rb') as f:
    # Load the association object
    association = pickle.load(f)

n_annotations = pd.read_csv('../Annotate/Batch1/annotated_calls_north_2021-11-04_02:00:02.csv')
s_annotations = pd.read_csv('../Annotate/Batch1/annotated_calls_south_2021-11-04_02:00:02.csv')

print(association.keys())
print(association['assoc_pair'].keys())
print(association['assoc'].keys())
print(association['metadata']['north'].keys())

# Load the metadata
fs = association['metadata']['north']['fs']
dx = association['metadata']['north']['dx']
n_selected_channel_m = association['metadata']['north']['selected_channels_m']
n_selected_channel = association['metadata']['north']['selected_channels']
s_selected_channel_m = association['metadata']['south']['selected_channels_m']
s_selected_channel = association['metadata']['south']['selected_channels']  

print(dx, n_selected_channel_m, s_selected_channel_m)   


tp = 0 # True positive
fp = 0 # False positive
tn = 0 # True negative
fn = 0 # False negative

nhf_pairs = association['assoc_pair']['north']['hf']
nlf_pairs = association['assoc_pair']['north']['lf']
nhf_assoc = association['assoc']['north']['hf']
nlf_assoc = association['assoc']['north']['lf']

shf_pairs = association['assoc_pair']['south']['hf']
slf_pairs = association['assoc_pair']['south']['lf']
shf_assoc = association['assoc']['south']['hf']
slf_assoc = association['assoc']['south']['lf']

nhf_annotations = n_annotations[(n_annotations['call_type'] == 'HF') & (n_annotations['call_id'] == 2)]
# print(nhf_annotations)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)

plt.scatter(nhf_pairs[0][1] / fs, n_selected_channel_m[0] + nhf_pairs[0][0] * dx * n_selected_channel[2], c='blue', s=1, alpha=0.5)
plt.scatter(nhf_annotations['time'], nhf_annotations['dist'])
plt.title('North')

tol = 0.1
# Check if the annotation is within the tolerance of the association   
for t, d in zip(nhf_annotations['time'], nhf_annotations['dist']):
    if  np.any(np.abs(nhf_pairs[0][1] / fs - t) < tol) and \
        np.any(np.abs(n_selected_channel_m[0] + nhf_pairs[0][0] * dx * n_selected_channel[2] - d) < tol):
        print('True positive')

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict

# First batch of data
with open('../denoised_data/Batch_3/association_2021-11-04_16:08:32.pkl', 'rb') as f:
    # Load the association object
    association = pickle.load(f)

# Load annotations
n_annotations = pd.read_csv('../Annotate/Batch3/annotated_calls_north_2021-11-04_16:08:32.csv')
s_annotations = pd.read_csv('../Annotate/Batch3/annotated_calls_south_2021-11-04_16:08:32.csv')

# Second batch of data 
# Load automated association data
# with open('../out/association_2021-11-03_22:09:42.pkl', 'rb') as f:
#     association = pickle.load(f)

# # Load annotations
# n_annotations = pd.read_csv('../Annotate/annotated_calls_north_2021-11-03_22:09:42.csv')
# s_annotations = pd.read_csv('../Annotate/annotated_calls_south_2021-11-03_22:09:42.csv')

# Metadata
fs = association['metadata']['north']['fs']
dx = association['metadata']['north']['dx']
n_selected_channel = association['metadata']['north']['selected_channels']
n_selected_channel_m = association['metadata']['north']['selected_channels_m']
s_selected_channel = association['metadata']['south']['selected_channels']
s_selected_channel_m = association['metadata']['south']['selected_channels_m']

# Association
assoc_pair = association['assoc_pair']
assoc = association['assoc']

# Parameters
time_tol = 0.5  # seconds
dist_tol = 20   # meters

# Helper
def get_time_dist(pairs, fs, dx, selected_channels, selected_channel_m):
    times = pairs[1] / fs
    dists = selected_channel_m[0] + pairs[0] * dx * selected_channels[2]
    return np.vstack([times, dists]).T

confusion = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0})
totals = {'TP': 0, 'FP': 0, 'FN': 0}

fig, axs = plt.subplots(2, 2, figsize=(15, 10))
axs = axs.flatten()
plot_idx = 0

for array, annots, sel_ch, sel_ch_m in [
    ('north', n_annotations, n_selected_channel, n_selected_channel_m),
    ('south', s_annotations, s_selected_channel, s_selected_channel_m)
]:
    for call_type in ['hf', 'lf']:
        key = f'{array}_{call_type.upper()}'
        a_all = annots[annots['call_type'].str.upper() == call_type.upper()]

        # Get all detection points for this call type and array
        pairs_list = assoc_pair[array][call_type]
        single_list = assoc[array][call_type]
        all_pairs = [get_time_dist(pairs, fs, dx, sel_ch, sel_ch_m) for pairs in pairs_list]
        all_single = [get_time_dist(single, fs, dx, sel_ch, sel_ch_m) for single in single_list]
        detected = np.vstack(all_pairs + all_single) if all_pairs else np.empty((0, 2))

        TP, FP, FN = 0, 0, 0

        # For plotting all annotation points (not just last one)
        all_annotated_points = []

        # Loop through each annotated call (cluster of points with a shared call_id)
        for call_id in a_all['call_id'].unique():
            a_call = a_all[a_all['call_id'] == call_id]
            if len(a_call) == 0:
                continue
            annotated_pts = a_call[['time', 'dist']].values
            all_annotated_points.append(annotated_pts)

            # Match if *any* annotation point is close to a detection
            matched = False
            for t_ann, d_ann in annotated_pts:
                if len(detected) == 0:
                    break
                dt = np.abs(detected[:, 0] - t_ann)
                dd = np.abs(detected[:, 1] - d_ann)
                if np.any((dt < time_tol) & (dd < dist_tol)):
                    matched = True
                    break

            if matched:
                TP += 1
            else:
                FN += 1

        # TODO: Solve the case where the pairing is not done in the annotation, then leading to negative FP
        FP = len(pairs_list) + len(single_list) - TP
        if FP < 0:
            FP = 0  # Ensure FP is not negative

        # Update confusion dict
        confusion[key] = {'TP': TP, 'FP': FP, 'FN': FN}
        totals['TP'] += TP
        totals['FP'] += FP
        totals['FN'] += FN

        # Plot
        ax = axs[plot_idx]
        plot_idx += 1
        if len(detected) > 0:
            ax.scatter(detected[:, 0], detected[:, 1], c='tab:blue', s=10, label='Detections')
        if len(all_annotated_points) > 0:
            annotated_all = np.vstack(all_annotated_points)
            ax.scatter(annotated_all[:, 0], annotated_all[:, 1], c='tab:orange', s=30, marker='x', label='Annotations')

        ax.set_title(f'{key} — TP: {TP}, FP: {FP}, FN: {FN}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Distance (m)')
        ax.set_xlim(0, 70)
        ax.grid()
        ax.legend()

plt.tight_layout()
plt.show()

# Print per-class confusion
print(f"\n{'Array_Type':<15} {'TP':>5} {'FP':>5} {'FN':>5}")
for k, v in confusion.items():
    print(f"{k:<15} {v['TP']:>5} {v['FP']:>5} {v['FN']:>5}")

# Print totals
print(f"\nTotal           {totals['TP']:>5} {totals['FP']:>5} {totals['FN']:>5}")

