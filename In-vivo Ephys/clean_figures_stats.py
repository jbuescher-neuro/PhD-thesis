# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 11:53:44 2025

@author: julch
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import spikeinterface.full as si
import seaborn as sns
from scipy import stats
sns.set_style("white")

#43 units 8 responding to light full protocol
base_folder1 = Path('E:/PhD/2. In-vivo Ephys Data/Stim1_B6J-8464/2024-09-13_14-58-00/Record Node 115')
exp_path1= base_folder1 / 'Exp4_excl'


#40 units 3 light responsive, full protocol
base_folder2 = Path(r'E:/PhD/2. In-vivo Ephys Data/JB_B6J-8636/2024-10-16_14-10-22/Record Node 104')
exp_path2= base_folder2 / r'Exp1'

combined_path= Path('E:/PhD/2. In-vivo Ephys Data/result/')

template= pd.read_csv(combined_path/'template1.csv')
combined_fid= pd.read_csv(combined_path+'fidelity1_fidtest2.csv')

#%%plot ephys metrics: peak to valley, firing range

#template.peak_to_valley= template.peak_to_valley*1000
sns.set_style("whitegrid", {'axes.grid' : False})


plt.figure(dpi=250)
ax=plt.subplot()
sns.scatterplot(data= template, y='peak_to_valley', x= 'base_FR', hue= 'light_response', palette= {'responsive':'darkorange', 'unresponsive':'grey'}, alpha=0.8,)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.ylabel('peak to valley [ms]')
plt.xlabel('Baseline spike rate [Hz]')
plt.tight_layout()
plt.savefig(combined_path/'1peak_freq.svg')
plt.show()


plt.figure(dpi=250)
ax=plt.subplot()
sns.scatterplot(data= template, y='FR_postPrim4', x= 'base_FR', hue= 'light_response', palette= {'responsive':'darkorange', 'unresponsive':'grey'}, alpha=0.8,)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.ylabel('Firing Rate 20 min post Priming [HZ]')
plt.xlabel('Baseline Firing Rate [Hz]')
plt.tight_layout()
bbox = plt.gca().get_window_extent()
ratio = bbox.width/bbox.height
plt.xlim(xmin=plt.ylim()[0],xmax=plt.ylim()[1]*ratio)
plt.axline((0, 0), slope=1, color='k', transform=plt.gca().transAxes, ls= '--')
#plt.savefig(combined_path/'1basevspostprim.svg')
plt.show()


#%%Plot Latencies, Firing Fidelity

plt.figure(dpi=250)
ax=plt.subplot()
sns.scatterplot(data= combined_fid, x='MeanLatency', y= 'LatencyVariance', c='coral', alpha=0.8,)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.ylabel('LantencyVariance [ms]')
plt.xlabel('Mean Latency [ms]')
plt.title('Latency vs Latency Variance in Fidelity Test2')
plt.tight_layout()
plt.savefig(combined_path/'latvsvariance.tif')
plt.show()

plt.figure(dpi=250)
ax=plt.subplot()
sns.scatterplot(data= combined_fid, y='FiringFidelity', x= 'LatencyVariance', c='coral', alpha=0.8,)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xlabel('LantencyVariance [ms]')
plt.ylabel('Firing Fidelity')
plt.title('Latency Variance vs Fidelity in Fidelity Test2')
plt.tight_layout()
plt.savefig(combined_path/'fidvsvariance.tif')
plt.show()

plt.figure(dpi=250)
ax= combined_fid.hist(column= 'MeanLatency', grid= False, color= 'coral', alpha=0.8)
for spine in ['top', 'right']:
    ax[0][0].spines[spine].set_visible(False)  # This removes the specified spines

# Set tick directions outward (optional, for cleaner look)
#ax[0][0].yaxis.set_tick_params(direction='out')
#ax[0][0].xaxis.set_tick_params(direction='out')

plt.xlabel('Lantency [ms]')
plt.ylabel('Count')
plt.title('Mean Latency in Fid2')
plt.xlim([0,15])
plt.savefig(combined_path/'1meanLat.svg')
plt.show()

#%%Plots over baselines figure firing rates

dif_long = pd.melt(template, 
                  id_vars=['light_response', 'si_unit_id'], 
                  value_vars=['delta_postResTest1',  'delta_postPrim1', 'delta_postPrim4', 'delta_postPrim7', 'delta_postResTest2'],
                  var_name= 'Condition',
                  value_name='Firing Rate')

plt.figure(dpi= 300, figsize=(10, 6))
ax=plt.subplot()
sns.lineplot(data=dif_long, y='Firing Rate', x= 'Condition',
             hue='light_response', palette={'responsive':'darkorange', 'unresponsive':'grey'},
    hue_order=['responsive', 'unresponsive'])

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.grid(False)
plt.ylabel('Difference in Firing Rate (Hz)')
plt.tight_layout()
#plt.savefig(Path(combined_path / 'diff_FR1.svg'))#exp_path/r'figures/FR_rec.svg')
plt.show()

# Reshape for seaborn
df_long = pd.melt(template, 
                  id_vars=['light_response', 'si_unit_id'], 
                  value_vars=['base_FR',  'FR_postFid1', 'FR_postPrim1', 'FR_postPrim4', 'FR_postPrim7', 'FR_postFid2'],
                  var_name= 'Condition',
                  value_name='Firing Rate')

plt.figure(dpi= 300, figsize=(10, 6))
ax=plt.subplot()
sns.boxplot(data=df_long, y='Firing Rate', x= 'Condition',
             hue='light_response', palette={'responsive':'darkorange', 'unresponsive':'grey'},
    hue_order=['responsive', 'unresponsive'], boxprops=dict(alpha=1))

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.grid(False)
plt.title('Firing Rates per Unit')
plt.ylabel('Firing Rate (Hz)')
plt.tight_layout()
#plt.savefig(combined_path/'FR_rec1.svg')
plt.show()

#%%Stats firing over recordings
from scipy.stats import shapiro
# Test normality within each group and condition
normality_results = []
for group in df_long.light_response.unique():
    group_data = df_long[df_long['light_response'] == group]
    for cond in group_data['Condition'].unique():
        values = group_data[group_data['Condition'] == cond]['Firing Rate'].dropna().values
        if len(values) >= 3:  # Shapiro needs at least 3 values
            stat, p = shapiro(values)
            normality_results.append((group, cond, stat, p))

# Display normality test results
for result in normality_results:
    group, cond, stat, p = result
    print(f"Shapiro test for {group}, {cond}: Statistic = {stat}, p-value = {p}")

#Stat for fr over time
from scipy.stats import friedmanchisquare,wilcoxon, mannwhitneyu
from statsmodels.stats.multitest import multipletests

df_long['Firing Rate'] = pd.to_numeric(df_long['Firing Rate'], errors='coerce')

# Perform the Friedman test within each group
results_friedman = []
groups = df_long['light_response'].unique()

for group in groups:
    group_data = df_long[df_long['light_response'] == group]
    # Perform the Friedman test for this group
    grouped = [group_data[group_data['Condition'] == cond]['Firing Rate'].values for cond in group_data['Condition'].unique()]
    stat, p = friedmanchisquare(*grouped)
    results_friedman.append((group, stat, p))

# Display Friedman test results
for group, stat, p in results_friedman:
    print(f"Friedman test for {group} group: Statistic = {stat}, p-value = {p}")

# Post-hoc tests (Wilcoxon signed-rank test) for significant results
post_hoc_results, raw_pvals = [], []

for group in groups:
    group_data = df_long[df_long['light_response'] == group]
    conditions  = group_data['Condition'].unique()
    
    for i, cond1 in enumerate(conditions):
        for cond2 in conditions[i+1:]:
            d1 = group_data[group_data['Condition'] == cond1]['Firing Rate']
            d2 = group_data[group_data['Condition'] == cond2]['Firing Rate']
            stat, p = wilcoxon(d1, d2)          # paired, non‑parametric
            post_hoc_results.append([group, cond1, cond2, stat, p])
            raw_pvals.append(p)

# Holm (FWER) – change to method='fdr_bh' for FDR
reject, p_corr, _, _ = multipletests(raw_pvals, alpha=0.05,
                                     method='holm')

# Attach corrected p‑values and decision flag
for k, row in enumerate(post_hoc_results):
    row.extend([p_corr[k], reject[k]])

print("Post‑hoc Wilcoxon tests with Holm correction")
for group, c1, c2, stat, p_raw, p_adj, sig in post_hoc_results:
    print(f"{group}: {c1} vs {c2} | W={stat:.2f}, "
          f"p_raw={p_raw:.4g}, p_Holm={p_adj:.4g}, reject={sig}")

# ------------------------------------------------------------
# 2. Between‑group Mann‑Whitney tests  +  Holm correction
# ------------------------------------------------------------
between_group_results, raw_pvals_BG = [], []

for tp in conditions:
    d_resp = df_long[(df_long['light_response']=='responsive') &
                     (df_long['Condition']==tp)]['Firing Rate']
    d_un   = df_long[(df_long['light_response']=='unresponsive') &
                     (df_long['Condition']==tp)]['Firing Rate']
    stat, p = mannwhitneyu(d_resp, d_un, alternative='two-sided')
    between_group_results.append([tp, stat, p])
    raw_pvals_BG.append(p)

reject_BG, p_corr_BG, _, _ = multipletests(raw_pvals_BG, alpha=0.05,
                                           method='holm')

for k, row in enumerate(between_group_results):
    row.extend([p_corr_BG[k], reject_BG[k]])

print("\nBetween‑group Mann‑Whitney tests with Holm correction")
for tp, stat, p_raw, p_adj, sig in between_group_results:
    print(f"{tp}: U={stat:.2f}, p_raw={p_raw:.4g}, "
          f"p_Holm={p_adj:.4g}, reject={sig}")
# Plot the boxplot
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_long, x='Condition', y='Firing Rate', 
            hue='light_response', estimator= None, units= 'si_unit_id',palette={'responsive':'darkorange', 'unresponsive':'grey'},
            hue_order=['responsive', 'unresponsive'])
plt.title('Firing Rates per Unit over Time')
plt.ylabel('Firing Rate (Hz)')
plt.tight_layout()
plt.show()
#%%Boxplot difference 20 min post vs baseline

# 20 min post vs baseline
plt.figure(dpi=250)
ax=plt.subplot()
sns.boxplot(data= template, y='FR_delta', x='light_response', hue= 'light_response', palette= {'responsive':'darkorange', 'unresponsive':'grey'},  boxprops=dict(alpha=.8))
sns.stripplot(data= template, y='FR_delta', x='light_response',hue= 'light_response', palette= {'responsive':'darkorange', 'unresponsive':'grey'}, jitter= True)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.ylabel('Firing Range range [HZ]')
plt.ylim([-10, 35])
plt.title('Base vs 20 min post')
plt.tight_layout()
plt.savefig(combined_path/'1box_base_postprim_range.svg')
plt.show()

#Stats FR delta Base vs 20 min post

# Define groups
unresp = template.FR_delta[template.light_response == 'unresponsive']
resp = template.FR_delta[template.light_response == 'responsive']

# Test for normality
shapiro_unresp = stats.shapiro(unresp)
shapiro_resp = stats.shapiro(resp)

print(f"Shapiro test unresponsive: W={shapiro_unresp.statistic:.3f}, p={shapiro_unresp.pvalue:.3f}")
print(f"Shapiro test responsive: W={shapiro_resp.statistic:.3f}, p={shapiro_resp.pvalue:.3f}")

alpha = 0.05

# Between-group test
if shapiro_unresp.pvalue > alpha and shapiro_resp.pvalue > alpha:
    # Use parametric test
    t_stat, p_value = stats.ttest_ind(unresp, resp)
    print(f"\nTwo-sample t-test: t={t_stat:.3f}, p={p_value:.3f}")
    if p_value < alpha:
        print("Reject the null hypothesis; significant difference between groups.")
    else:
        print("Fail to reject the null hypothesis; no significant difference between groups.")
else:
    # Use non-parametric test
    u_stat, p_value = stats.mannwhitneyu(unresp, resp, alternative='two-sided')
    print(f"\nMann–Whitney U test: U={u_stat:.3f}, p={p_value:.3f}")
    if p_value < alpha:
        print("Reject the null hypothesis; significant difference between groups.")
    else:
        print("Fail to reject the null hypothesis; no significant difference between groups.")

# One-sample tests against 0
print("\nOne-sample tests against 0:")

# Unresponsive group
if shapiro_unresp.pvalue > alpha:
    t_unresp, p_unresp = stats.ttest_1samp(unresp, 0)
    print(f"Unresponsive group (t-test): t={t_unresp:.3f}, p={p_unresp:.3f}")
else:
    w_unresp, p_unresp = stats.wilcoxon(unresp)
    print(f"Unresponsive group (Wilcoxon): W={w_unresp:.3f}, p={p_unresp:.3f}")

# Interpretation
if p_unresp < alpha:
    print("Reject H0; unresponsive group significantly different from 0.")
else:
    print("Fail to reject H0; unresponsive group not significantly different from 0.")

# Responsive group
if shapiro_resp.pvalue > alpha:
    t_resp, p_resp = stats.ttest_1samp(resp, 0)
    print(f"\nResponsive group (t-test): t={t_resp:.3f}, p={p_resp:.3f}")
else:
    w_resp, p_resp = stats.wilcoxon(resp)
    print(f"\nResponsive group (Wilcoxon): W={w_resp:.3f}, p={p_resp:.3f}")

# Interpretation
if p_resp < alpha:
    print("Reject H0; responsive group significantly different from 0.")
else:
    print("Fail to reject H0; responsive group not significantly different from 0.")


#%%Boxplot+ stats diff Light responsiveness Tests

plt.figure(dpi=250)
ax=plt.subplot()
sns.boxplot(data= template, y='FR_delta_fid', x='light_response', hue= 'light_response', palette= {'responsive':'darkorange', 'unresponsive':'grey'},  boxprops=dict(alpha=.8))
sns.stripplot(data= template, y='FR_delta_fid', x='light_response',hue= 'light_response', palette= {'responsive':'darkorange', 'unresponsive':'grey'}, jitter= True)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.ylabel('Firing Range range between fidelity tests [HZ]')
plt.ylim([-10, 35])
plt.tight_layout()
plt.savefig(combined_path/'1fid1vsfid2.svg')
plt.show()

# Define groups
unresp = template.FR_delta_fid[template.light_response == 'unresponsive']
resp = template.FR_delta_fid[template.light_response == 'responsive']

# Test for normality
shapiro_unresp = stats.shapiro(unresp)
shapiro_resp = stats.shapiro(resp)

print(f"Shapiro test unresponsive between Fid: W={shapiro_unresp.statistic:.3f}, p={shapiro_unresp.pvalue:.3f}")
print(f"Shapiro test responsive between Fid: W={shapiro_resp.statistic:.3f}, p={shapiro_resp.pvalue:.3f}")

alpha = 0.05

# Between-group test
if shapiro_unresp.pvalue > alpha and shapiro_resp.pvalue > alpha:
    # Use parametric test
    t_stat, p_value = stats.ttest_ind(unresp, resp)
    print(f"\nTwo-sample t-test: t={t_stat:.3f}, p={p_value:.3f}")
    if p_value < alpha:
        print("Reject the null hypothesis; significant difference between groups between post fid.")
    else:
        print("Fail to reject the null hypothesis; no significant difference between groups between post fid.")
else:
    # Use non-parametric test
    u_stat, p_value = stats.mannwhitneyu(unresp, resp, alternative='two-sided')
    print(f"\nMann–Whitney U test: U={u_stat:.3f}, p={p_value:.3f}")
    if p_value < alpha:
        print("Reject the null hypothesis; significant difference between groups between post fid.")
    else:
        print("Fail to reject the null hypothesis; no significant difference between groups between post fid.")

# One-sample tests against 0
print("\nOne-sample tests against 0:")

# Unresponsive group
if shapiro_unresp.pvalue > alpha:
    t_unresp, p_unresp = stats.ttest_1samp(unresp, 0)
    print(f"Unresponsive group (t-test): t={t_unresp:.3f}, p={p_unresp:.3f}")
else:
    w_unresp, p_unresp = stats.wilcoxon(unresp)
    print(f"Unresponsive group (Wilcoxon): W={w_unresp:.3f}, p={p_unresp:.3f}")

# Interpretation
if p_unresp < alpha:
    print("Reject H0; unresponsive group significantly different from 0 between post fid.")
else:
    print("Fail to reject H0; unresponsive group not significantly different from 0 between post fid.")

# Responsive group
if shapiro_resp.pvalue > alpha:
    t_resp, p_resp = stats.ttest_1samp(resp, 0)
    print(f"\nResponsive group (t-test): t={t_resp:.3f}, p={p_resp:.3f}")
else:
    w_resp, p_resp = stats.wilcoxon(resp)
    print(f"\nResponsive group (Wilcoxon): W={w_resp:.3f}, p={p_resp:.3f}")

# Interpretation
if p_resp < alpha:
    print("Reject H0; responsive group significantly different from 0 between post fid.")
else:
    print("Fail to reject H0; responsive group not significantly different from 0 between post fid.")


#%%Latency heatmaps
combined_lat= pd.read_csv(combined_path/'lats_FidTest2.csv')
combined_lat2= pd.read_csv(combined_path/'lats_FidTest1.csv')
#Heatmap of units latencies

latency_cols = [c for c in combined_lat.columns if c.startswith('Latency_')]
latency_cols2 = [c for c in combined_lat2.columns if c.startswith('Latency_')]

# reshape wide → long
long = combined_lat.melt(id_vars=['Neuron_ID'], value_vars=latency_cols,
                         var_name='Trial', value_name='Latency').dropna()
long2 = combined_lat2.melt(id_vars=['Neuron_ID'], value_vars=latency_cols2,
                           var_name='Trial', value_name='Latency').dropna()

# keep only latencies ≤ 100 ms
long = long[long['Latency'] <= 100]
long2 = long2[long2['Latency'] <= 100]

# define bins (2 ms steps)
bins = np.arange(0, 101, 2)

def heat_from_wide(df, unit_col='Neuron_ID', latency_prefix='Latency_', max_latency=100, bin_width=2):
    latency_cols = [c for c in df.columns if c.startswith(latency_prefix)]
    bins = np.arange(0, max_latency + bin_width, bin_width)
    heat = []
    units = df[unit_col].tolist()
    for _, row in df.iterrows():
        vals = row[latency_cols].values.astype(float)
        vals = vals[~np.isnan(vals)]
        vals = vals[vals <= max_latency]
        counts, _ = np.histogram(vals, bins=bins)
        heat.append(counts)
    return np.array(heat), units, bins

heat1, units, bins = heat_from_wide(combined_lat, 'Neuron_ID')
heat2, units2, _   = heat_from_wide(combined_lat2, 'Neuron_ID')

# if unit order differs, align heat2 to units:
if units != units2:
    idx_map = {u:i for i,u in enumerate(units2)}
    heat2 = np.vstack([heat2[idx_map[u]] if u in idx_map else np.zeros(heat2.shape[1],dtype=int) for u in units])

# --- plot ---
bin_width = 2
num_bins = heat1.shape[1]
# bin centers for labeling
bin_centers = np.arange(0, num_bins) * bin_width + bin_width/2

vmax = max(heat1.max(), heat2.max())
max_rows = max(heat1.shape[0], heat2.shape[0])

def pad_heat(data, target_rows):
    pad_rows = target_rows - data.shape[0]
    if pad_rows > 0:
        return np.pad(data, ((0, pad_rows), (0,0)), mode='constant')
    return data

heat1_padded = pad_heat(heat1, max_rows)
heat2_padded = pad_heat(heat2, max_rows)


# --- responsive-first ordering  ---
labels = (combined_lat[['Neuron_ID','light_response']]
          .drop_duplicates('Neuron_ID')
          .assign(light_response=lambda d: d['light_response'].astype(str).str.strip().str.lower()))

order_df = pd.DataFrame({'unit': units}).merge(
    labels, left_on='unit', right_on='Neuron_ID', how='left'
)
order_df['resp_sort'] = (order_df['light_response'] == 'responsive').astype(int)

# stable sort: responsive (1) first, preserve original order within groups
sorted_idx = np.argsort(-order_df['resp_sort'].to_numpy(), kind='stable')

heat1_sorted = heat1_padded[sorted_idx, :]
heat2_sorted = heat2_padded[sorted_idx, :]
units_sorted = [units[i] for i in sorted_idx]

# Boolean array of which units are responsive in the *sorted* order
resp_sorted = (order_df['resp_sort'].to_numpy()[sorted_idx] == 1)

# Now define masks for heatmap plotting
resp_mask2d   = ~resp_sorted[:, None].repeat(num_bins, axis=1)   # mask out unresponsive → leaves responsive
unresp_mask2d =  resp_sorted[:, None].repeat(num_bins, axis=1)   # mask out responsive → leaves unresponsive


from matplotlib import cm, colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

# assume: heat1_sorted, heat2_sorted shape = (n_units, num_bins)
n_units, num_bins = heat1_sorted.shape
vmax = max(heat1_sorted.max(), heat2_sorted.max())
vmin = 0

fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

for ax, data, title in zip(axes, [heat2_sorted, heat1_sorted],
                           ["Light Responsiveness Test 1", "Light Responsiveness Test 2"]):
    # Color responsive units red (only those that have labels)
   
    resp_mask2d = ~resp_sorted[:, None].repeat(num_bins, axis=1)
    unresp_mask2d = resp_sorted[:, None].repeat(num_bins, axis=1)

    # make masked arrays
    data_resp = np.ma.array(data, mask=resp_mask2d)
    data_unresp = np.ma.array(data, mask=unresp_mask2d)

    # plot responsive (Reds) and unresponsive (Greys)
    ax.imshow(data_resp, cmap="Reds", vmin=vmin, vmax=vmax,
              aspect='auto', origin='upper', interpolation='nearest')
    ax.imshow(data_unresp, cmap="Greys", vmin=vmin, vmax=vmax,
              aspect='auto', origin='upper', interpolation='nearest')

    ax.set_title(title)

     # --- add colorbars outside each subplot ---
from mpl_toolkits.axes_grid1 import make_axes_locatable

# only for second heatmap
divider = make_axes_locatable(axes[1])

cax_resp = divider.append_axes("right", size="5%", pad=0.05)
mappable_resp = cm.ScalarMappable(norm=colors.Normalize(vmin=vmin, vmax=vmax), cmap="Reds")
fig.colorbar(mappable_resp, cax=cax_resp, label="Spike count (responsive)")

cax_unresp = divider.append_axes("right", size="5%", pad=0.15)  # slightly offset to avoid overlap
mappable_unresp = cm.ScalarMappable(norm=colors.Normalize(vmin=vmin, vmax=vmax), cmap="Greys")
fig.colorbar(mappable_unresp, cax=cax_unresp, label="Spike count (unresponsive)")




# X ticks
step_bins = 10
xtick_locs = np.arange(0, num_bins, step_bins)
xtick_labels = (bin_centers[xtick_locs]).astype(int)

for ax in axes:
    ax.set_xlabel("Latency (ms)")
    ax.set_xticks(xtick_locs)
    ax.set_xticklabels(xtick_labels)

    # Y ticks
    step = 5
    yticks = np.arange(0, n_units, step)
    ylabels = [str(i+1) for i in range(0, n_units, step)]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)

    # horizontal separator line
    n_resp = int(resp_sorted.sum())
    ax.axhline(n_resp - 0.5, color='black', linewidth=1.2)

plt.tight_layout()
plt.savefig(combined_path/'1latency_heat.svg', dpi=300, bbox_inches='tight')
plt.show()
#%%Box plots not used
df = pd.melt(template, 
                  id_vars=['light_response', 'si_unit_id'], 
                  value_vars=['base_FR',  'FR_postPrim4'],
                  var_name= 'Condition',
                  value_name='Firing Rate')

plt.figure(dpi=300, figsize=(10, 6))
df['Condition'] = df['Condition'].astype('category')

df['x_num'] = df['Condition'].cat.codes
df['x_num'] = df['x_num'].map({0: 1, 1: 0})
hue_offset = {'responsive': -0.2, 'unresponsive': 0.2}
df['x_dodge'] = df['x_num'] + df['light_response'].map(hue_offset)

ax = plt.subplot()
"""
# Lineplot with x_dodge (offset numeric positions)
sns.lineplot(
    data=df, x='x_dodge', y='Firing Rate',
    units='si_unit_id', hue='light_response',
    estimator=None, alpha=0.5,
    palette={'responsive': 'darkorange', 'unresponsive': 'grey'},
    hue_order=['responsive', 'unresponsive'],
    marker='o', linestyle='-',
    markeredgecolor='black', markeredgewidth=0.8,
    ax=ax
)
"""
# Boxplot with x_num (numeric categorical code)
sns.boxplot(
    data=df, y='Firing Rate', x='x_num',
    hue='light_response', palette={'responsive':'darkorange', 'unresponsive':'grey'},
    hue_order=['responsive', 'unresponsive'],
    boxprops=dict(alpha=1),
    ax=ax
)

# Set legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2], title='Light response')

# Set ticks at integer positions (x_num)
ax.set_xticks(df['x_num'].unique())
# Set tick labels to category names, preserving the original category order
ax.set_xticklabels(['baseline', '20 min PostPriming'])

# Clean up plot style
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.grid(False)

plt.ylabel('Firing Rate (Hz)')
plt.xlabel('Condition')
#plt.savefig(combined_path/'1BOXbasevsPostprim.svg')
plt.tight_layout()
plt.show()

"""
#not usedBaseline vs Stim Firing rate
plt.figure(dpi=300)
ax=plt.subplot()
sns.scatterplot(data= template, y='FR_postPrim4', x= 'base_FR', hue= 'light_response', palette={'responsive':'darkorange', 'unresponsive':'grey'})

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
#ax.grid(False)
plt.title('Baseline vs Post Prim4 firing')
bbox = plt.gca().get_window_extent()
ratio = bbox.width/bbox.height
plt.xlim(xmin=plt.ylim()[0],xmax=plt.ylim()[1]*ratio)
plt.axline((0, 0), slope=1, color='k', transform=plt.gca().transAxes, ls= '--')
plt.savefig(combined_path /'basevsPostprim.svg')#r'basevsPostPrim.tif')
plt.tight_layout()
plt.show()"""
