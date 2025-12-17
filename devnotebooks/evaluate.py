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

# +
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import os
from collections import defaultdict
from glob import glob 
from tqdm import tqdm

plt.rcParams['font.size'] = 16

# +
# Directory and batch setup
batches = ['Batch_1', 'Batch_2', 'Batch_3', 'Batch_4', 'Batch_5']
# automated associations folder: 
data_root = '../denoised_data'
method = 'FarWin'  # Options: 'Baseline', 'FarWin', 'Gabor_Farwin'
# manual annotations folder:
annot_root = '../Annotate'

# Parameters
time_tol = 0.5  # seconds
dist_tol = 20   # meters

# +
# Batch1 2021-11-04_02:00:02
# Batch2 2021-11-03_22:10:42
# Batch3 2021-11-04_16:15:12
# Batch4 2021-11-04_08:00:02
# Batch5 2021-11-04_12:09:12

batch = '4'
timestamp = '2021-11-04_08:06:42'

def load_data(batch, timestamp):
    """Load the data for a specific batch and timestamp."""
    data_file = f'{data_root}/Batch_{batch}/{method}/association_{timestamp}.pkl'
    with open(data_file, 'rb') as f:
        assoc = pickle.load(f)
    
    n_annot = pd.read_csv(f'{annot_root}/Batch{batch}/annotated_calls_north_{timestamp}.csv')
    s_annot = pd.read_csv(f'{annot_root}/Batch{batch}/annotated_calls_south_{timestamp}.csv')
    return assoc, n_annot, s_annot


def get_time_dist(pairs, fs, dx, selected_channels, selected_channel_m):
    times = pairs[1] / fs
    dists = selected_channel_m[0] + pairs[0] * dx * selected_channels[2]
    return np.vstack([times, dists]).T



# +
association, n_annotations, s_annotations = load_data(batch, timestamp)

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

confusion = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0})
totals = {'TP': 0, 'FP': 0, 'FN': 0}

fig, axs = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
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
        all_pairs = [get_time_dist(pairs, fs, dx, sel_ch, sel_ch_m) for pairs in pairs_list if pairs[0].size >= 500]
        all_single = [get_time_dist(single, fs, dx, sel_ch, sel_ch_m) for single in single_list if single[0].size >= 500]
        detected = np.vstack(all_pairs + all_single)  if all_pairs or all_single else np.empty((0, 2))

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
        if key.startswith('north'):
            ax.set_ylim(12000, 66000)
        elif key.startswith('south'):
            ax.set_ylim(12000, 95000)   
        ax.grid(alpha=0.8, linestyle='--')
        ax.legend()

# plt.tight_layout()
plt.show()

# Print per-class confusion
print(f"\n{'Array_Type':<15} {'TP':>5} {'FP':>5} {'FN':>5}")
for k, v in confusion.items():
    print(f"{k:<15} {v['TP']:>5} {v['FP']:>5} {v['FN']:>5}")

# Print totals
print(f"\nTotal           {totals['TP']:>5} {totals['FP']:>5} {totals['FN']:>5}")
plt.savefig(f'../figs/evaluation_{method}_{timestamp}_batch{batch}.pdf', bbox_inches=None, transparent=True)

# +
confusion = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0})
totals = {'TP': 0, 'FP': 0, 'FN': 0}
plot_figs = True

for batch in tqdm(batches, """  """desc=f'Processing annotated batches'):
    assoc_files = sorted(glob(f'{data_root}/{batch}/{method}/association_*.pkl'))
    # Dictionary for batch"""  """-level confusion
    total_batch = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0})

    for assoc_path in assoc_files:
        timestamp = os.path.basename(assoc_path).replace('association_', '').replace('.pkl', '')
        # print(f"Processing {batch} - {timestamp}")
        
        # Load association
        with open(assoc_path, 'rb') as f:
            association = pickle.load(f)

        # Load annotations
        try:
            n_annotations = pd.read_csv(f'{annot_root}/{batch.replace("Batch_", "Batch")}/annotated_calls_north_{timestamp}.csv')
            s_annotations = pd.read_csv(f'{annot_root}/{batch.replace("Batch_", "Batch")}/annotated_calls_south_{timestamp}.csv')
        except FileNotFoundError:
            print(f"Annotation missing for {timestamp}, skipping.")
            continue

        fs = association['metadata']['north']['fs']
        dx = association['metadata']['north']['dx']
        n_selected_channel = association['metadata']['north']['selected_channels']
        n_selected_channel_m = association['metadata']['north']['selected_channels_m']
        s_selected_channel = association['metadata']['south']['selected_channels']
        s_selected_channel_m = association['metadata']['south']['selected_channels_m']
        assoc_pair = association['assoc_pair']
        assoc = association['assoc']

        if plot_figs:
            fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=True, constrained_layout=True)
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
                detected = np.vstack(all_pairs + all_single)  if all_pairs or all_single else np.empty((0, 2))

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

                FP = len(pairs_list) + len(single_list) - TP
                if FP < 0:
                    FP = 0  # Ensure FP is not negative

                # Key-level confusion update
                confusion[key]['TP'] += TP
                confusion[key]['FP'] += FP
                confusion[key]['FN'] += FN

                # Batch-level confusion update
                confusion[batch]['TP'] += TP
                confusion[batch]['FP'] += FP
                confusion[batch]['FN'] += FN

                # Update totals
                totals['TP'] += TP
                totals['FP'] += FP
                totals['FN'] += FN
                
                if plot_figs:
                    # Plot
                    ax = axs[plot_idx]
                    plot_idx += 1
                    if len(detected) > 0:
                        ax.scatter(detected[:, 0], detected[:, 1], c='tab:blue', s=10, label='Detections')
                    if len(all_annotated_points) > 0:
                        annotated_all = np.vstack(all_annotated_points)
                        ax.scatter(annotated_all[:, 0], annotated_all[:, 1], c='tab:orange', s=30, marker='x', label='Annotations')

                    # Create legend entries even if no data is plotted
                    from matplotlib.lines import Line2D
                    legend_elements = [
                        Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:blue', markersize=6, label='Detections'),
                        Line2D([0], [0], marker='x', color='tab:orange', linestyle='None', markersize=8, label='Annotations')
                    ]

                    ax.set_title(f'{key} — TP: {TP}, FP: {FP}, FN: {FN}')
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Distance (m)')
                    ax.set_xlim(0, 70)
                    if key.startswith('north'):
                        ax.set_ylim(12000, 66000)
                    elif key.startswith('south'):
                        ax.set_ylim(12000, 95000)   
                    ax.grid(alpha=0.8, linestyle='--')
                    ax.legend(handles=legend_elements)

        if plot_figs:
            # Save the figures
            plt.savefig(f'../evaluation/{batch}/association_{timestamp}.png', dpi=300, bbox_inches='tight')
            plt.close()

# +
# Final report
print(f"\n{'Array_Type':<15} {'TP':>5} {'FP':>5} {'FN':>5}")
for k, v in confusion.items():
    print(f"{k:<15} {v['TP']:>5} {v['FP']:>5} {v['FN']:>5}")

Precision = totals['TP'] / (totals['TP'] + totals['FP']) if (totals['TP'] + totals['FP']) > 0 else 0
Recall = totals['TP'] / (totals['TP'] + totals['FN']) if (totals['TP'] + totals['FN']) > 0 else 0
F1 = 2 * (Precision * Recall) / (Precision + Recall) if (Precision + Recall) > 0 else 0

print(f"\nOverall Metrics:")
print(f"Precision: {Precision:.3f}")
print(f"Recall: {Recall:.3f}")
print(f"F1 Score: {F1:.3f}")

for batch in batches:
    if batch in confusion:
        batch_totals = confusion[batch]
        batch_precision = batch_totals['TP'] / (batch_totals['TP'] + batch_totals['FP']) if (batch_totals['TP'] + batch_totals['FP']) > 0 else 0
        batch_recall = batch_totals['TP'] / (batch_totals['TP'] + batch_totals['FN']) if (batch_totals['TP'] + batch_totals['FN']) > 0 else 0
        batch_f1 = 2 * (batch_precision * batch_recall) / (batch_precision + batch_recall) if (batch_precision + batch_recall) > 0 else 0

        print(f"\nBatch {batch} Metrics:")
        print(f"Precision: {batch_precision:.3f}")
        print(f"Recall: {batch_recall:.3f}")
        print(f"F1 Score: {batch_f1:.3f}")

# +
# Plot the statistics per batch and overall
fig, axs = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)

# Define colors for consistency
batch_color = 'steelblue'
overall_color = 'darkorange'

# Collect data for plotting
batch_names = []
precision_vals = []
recall_vals = []
f1_vals = []

for batch in batches:
    if batch in confusion:
        batch_totals = confusion[batch]
        precision = batch_totals['TP'] / (batch_totals['TP'] + batch_totals['FP']) if (batch_totals['TP'] + batch_totals['FP']) > 0 else 0
        recall = batch_totals['TP'] / (batch_totals['TP'] + batch_totals['FN']) if (batch_totals['TP'] + batch_totals['FN']) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        batch_names.append(batch)
        precision_vals.append(precision)
        recall_vals.append(recall)
        f1_vals.append(f1)

# Add overall statistics
batch_names.append('Overall')
precision_vals.append(Precision)
recall_vals.append(Recall)
f1_vals.append(F1)

# Create x positions
x_pos = range(len(batch_names))

# Plot bars with improved styling
bars1 = axs[0].bar(x_pos, precision_vals, color=[batch_color] * (len(batch_names)-1) + [overall_color], 
                   alpha=0.8, edgecolor='white', linewidth=1.5)
bars2 = axs[1].bar(x_pos, recall_vals, color=[batch_color] * (len(batch_names)-1) + [overall_color], 
                   alpha=0.8, edgecolor='white', linewidth=1.5)
bars3 = axs[2].bar(x_pos, f1_vals, color=[batch_color] * (len(batch_names)-1) + [overall_color], 
                   alpha=0.8, edgecolor='white', linewidth=1.5)

# Add value labels on top of bars
for ax, vals, bars in zip(axs, [precision_vals, recall_vals, f1_vals], [bars1, bars2, bars3]):
    for bar, val in zip(bars, vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

# Customize axes
titles = ['Precision', 'Recall', 'F1 Score']
for i, (ax, title) in enumerate(zip(axs, titles)):
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(batch_names, rotation=45, ha='right')
    ax.set_ylim(0, 1.1)  # Set consistent y-axis limits
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylabel('Score', fontsize=12)
    
    # Add horizontal reference lines
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax.axhline(y=0.8, color='green', linestyle=':', alpha=0.5, linewidth=1)

# # Add a main title
# fig.suptitle('Performance Metrics by Batch', fontsize=16, fontweight='bold', y=0.98)

# # Create a custom legend
# from matplotlib.patches import Patch
# legend_elements = [
#     Patch(facecolor=batch_color, alpha=0.8, label='Individual Batches'),
#     Patch(facecolor=overall_color, alpha=0.8, label='Overall Performance')
# ]
# fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.85))

# Add some styling improvements
plt.setp(axs, facecolor='#f8f9fa')
for ax in axs:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')

# plt.tight_layout()
plt.show()

