# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 13:33:04 2025

@author: julch
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import scipy.stats as stats
import seaborn as sns
import os
from scipy.stats import mannwhitneyu

#%%

def get_detect_normalize_and_classify_region(path, stim_dict, per_crebp, per_cfos):
    """Normalize by dividing through Control region: primary Motor Cortex,
    stim_dict needs to be a dictionary with mouse_ID and stimulated side"""
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    results = pd.DataFrame()
    
    for file in csv_files:
        file_path = os.path.join(path, file)
        print(file.split('_')[0])
        meas = pd.read_csv(file_path)

        # Cleanup and add relevant columns
        meas = meas.drop(columns=['Object ID', 'Name'])
        meas['MouseID'] = file.split('_')[0]
        meas['Image'] = meas.Image.str.split('.', n=1, expand=True)[0]
        meas['Side'] = meas.Parent.str.split(':', n=1, expand=True)[0]
        meas['Region'] = meas.Parent.str.split(':', n=1, expand=True)[1]
        
    
        # Assign stimulation side
        meas['Side'] = meas.apply(lambda row: 'Stim' if row['Side'] == stim_dict.get(row['MouseID'], 'Unknown') else 'NoStim', axis=1)
        
        # Append data
        results = pd.concat([results, meas], ignore_index=True)
        
    results['Region'] = results['Region'].str.strip()
    # Compute background (MOP mean intensity) for each MouseID, Image, Side

    background = results[results['Region'] == 'MOp'].groupby(['MouseID', 'Image', 'Side']).agg(
        cfos_background=('c-fos: Mean', 'median'),#, lambda x: x.quantile(1)),
        CREBP_background=('CREBP: Mean', 'median')#, lambda x: x.quantile(1))
    ).reset_index()
    print(background.head())
    
    # Merge background values back into the main results dataframe
    results = results.merge(background, on=['MouseID', 'Image', 'Side'], how='left')

    # Normalize by division
    results['cfos'] = results['c-fos: Mean'] / results['cfos_background']
    results['CREBP'] = results['CREBP: Mean'] / results['CREBP_background']
    
    results.loc[results['cfos'] < 0, 'cfos'] = 0
    results.loc[results['CREBP'] < 0, 'CREBP'] = 0
    results.loc[results['cfos'] > 10000, 'cfos'] = 0
    results.loc[results['CREBP'] > 10000, 'CREBP'] = 0
    
    
    print('Background normalization done!')

    # Classification using the quantile
    thresh_CREBP = results.groupby(['MouseID', 'Region', 'Image'])['CREBP'].transform(lambda x: x.quantile(per_crebp))
    thresh_cfos = results.groupby(['MouseID', 'Region', 'Image'])['cfos'].transform(lambda x: x.quantile(per_cfos))
    
    # Visualize effect of background normalization    
    sns.histplot(results['cfos'].dropna(), bins=70, label='division', kde=True)
    plt.axvline(thresh_cfos.median(), color='k', linestyle='--', label='Threhsold')
    plt.legend()
    plt.title("Effect of Background Normalization (c-Fos)")
    plt.show()

    sns.histplot(results['CREBP'].dropna(), bins=70, label='division', kde=True)
    plt.axvline(thresh_CREBP.median(), color='k', linestyle='--', label='Threhsold')
    plt.title("Effect of Background Normalization (CREBP)")
    plt.show()
    results['CREBP_10%'] = np.where(results['CREBP'] >= thresh_CREBP, 'CREBP', '')
    results['cfos_10%'] = np.where(results['cfos'] >= thresh_cfos, 'c-fos', '')

    results['Classification'] = np.where(
        (results['CREBP_10%'] != '') & (results['cfos_10%'] != ''), 
        'c-fos: CREBP',
        results['CREBP_10%'] + results['cfos_10%']
    )

    return results


def detect2anno(meas_crebp):
    """Create a table with counts per region for every image"""
    # Create empty results dataframe to store the final output
    results = pd.DataFrame(columns= ['MouseID', 'Image', 'Region', 'Side', 'Parent', 'Num Detections', 'Num c-fos', 'Num CREBP', 'Num c-fos: CREBP', 'rel_cfos', 'rel_CREBP', 'rel_colok'])
    
    # Get unique mouse and image combinations
    unique_mouseImage = meas_crebp[['MouseID', 'Image']].drop_duplicates()
    
    for _, row in unique_mouseImage.iterrows():
        mouse = row['MouseID']
        image = row['Image']
        print(mouse, image)
        
        # Filter the data for the current mouse and image
        image_data = meas_crebp[(meas_crebp['MouseID'] == mouse) & (meas_crebp['Image'] == image)]
        
        # Create a temporary table to store the data for the current mouse and image
        table = pd.DataFrame(columns= ['MouseID', 'Image', 'Region', 'Side', 'Parent', 'Num Detections', 'Num c-fos', 'Num CREBP', 'Num c-fos: CREBP', 'rel_cfos', 'rel_CREBP', 'rel_colok'])
        
        # Fill in the basic information (MouseID, Image, Region, Side, Parent)
        table[['MouseID', 'Image', 'Region', 'Side', 'Parent']] = image_data[['MouseID', 'Image', 'Region', 'Side', 'Parent']].drop_duplicates()
        
        # Initialize the columns for counts and relative counts
        table['Num Detections'] = 0
        table['Num c-fos'] = 0
        table['Num CREBP'] = 0
        table['Num c-fos: CREBP'] = 0
        table['rel_cfos'] = 0
        table['rel_CREBP'] = 0
        table['rel_colok'] = 0
        
        # Iterate through each region and compute counts
        for region in image_data['Parent'].unique():
            print(region)
            # Count the total number of detections for each region
            table.loc[table['Parent'] == region, 'Num Detections'] = image_data.loc[image_data['Parent'] == region, 'Image'].count()
            
            # Count occurrences of each classification
            table.loc[table['Parent'] == region, 'Num c-fos'] = image_data.loc[(image_data['Parent'] == region) & (image_data['Classification'].str.contains('c-fos', na=False)), 'Classification'].count()
            table.loc[table['Parent'] == region, 'Num CREBP'] = image_data.loc[(image_data['Parent'] == region) & (image_data['Classification'].str.contains('CREBP', na=False)), 'Classification'].count()
            table.loc[table['Parent'] == region, 'Num c-fos: CREBP'] = image_data.loc[(image_data['Parent'] == region) & (image_data['Classification'].str.contains('c-fos: CREBP', na=False)), 'Classification'].count()
        
        # Calculate relative counts
        table['rel_cfos'] = table['Num c-fos'] / table['Num Detections']
        table['rel_CREBP'] = table['Num CREBP'] / table['Num Detections']
        table['rel_colok'] = table['Num c-fos: CREBP'] / table['Num Detections']
        
        if 'CTX' not in table['Region'].values:
            # Sum the data for the 'CTX' region
            ctx_data = table.groupby(['MouseID', 'Image', 'Side'], as_index=False)[['Num Detections', 'Num c-fos', 'Num CREBP', 'Num c-fos: CREBP']].sum()
            ctx_data['Region'] = 'CTX'
            ctx_data['Parent'] = 'CTX'

            # Calculate relative counts for 'CTX'
            ctx_data['rel_cfos'] = ctx_data['Num c-fos'] / ctx_data['Num Detections']
            ctx_data['rel_CREBP'] = ctx_data['Num CREBP'] / ctx_data['Num Detections']
            ctx_data['rel_colok'] = ctx_data['Num c-fos: CREBP'] / ctx_data['Num Detections']
            
            # Append the 'CTX' data to the table
            table = pd.concat([table, ctx_data], ignore_index=True)
        # Append the current table to the results dataframe
        results = pd.concat([results, table], ignore_index=True)
    
    # Return the final results after processing all images and mice
    return results


def get_APs(path, meas):
    """take the AP positions of the fibers from file and only include relevant images"""
    ap = pd.read_excel(path + 'AP_slices.xlsx')
    meas['Region'] = meas['Region'].str.strip().str.upper()
    # Ensure consistent data types
    meas['MouseID'] = meas['MouseID'].astype(str).str.strip()
    meas['Image'] = meas['Image'].astype(str).str.strip()  # Remove leading/trailing spaces
    ap['MouseID'] = ap['MouseID'].astype(str).str.strip()
    ap['Image'] = ap['Image'].astype(str).str.strip()  # Remove leading/trailing spaces

    # Optionally: Ensure consistent case (e.g., lowercase for both)
    meas['Image'] = meas['Image'].str.lower()
    ap['Image'] = ap['Image'].str.lower()
    # Standardize the Image column by removing underscores (or adding underscores)
    meas['Image'] = meas['Image'].str.replace('_', '', regex=False)
    ap['Image'] = ap['Image'].str.replace('_', '', regex=False)
    
    # Merge AP coordinates into meas
    meas = meas.merge(ap[['MouseID', 'Image', 'AP']], on=['MouseID', 'Image'], how='left')
    
    return meas


#%% Average per mouse and Boxplots with error bars

meas= meas[~meas.MouseID.isin(['DHC-0945', 'DHC-0945.1', 'B6J-6629', 'B6J-8646', 'DHC-2260'])]
meas.rel_CREBP.loc[(meas.MouseID == 'DHC-0964')]= np.nan
meas = meas[~((meas.Region == 'PL') & (meas.AP <= 1.6) | ((meas.Region == 'ILA') & (meas.AP <= 1.6)))]
meas_AP = meas[(meas['AP'] >= 1.5) & (meas['AP'] <= 1.9)]
meas_noAP = meas[(meas['AP'] < 1.5) | (meas['AP'] > 1.9)]

agg_relative_counts = (meas_AP.groupby(['MouseID', 'Region', 'Side'])[['rel_cfos', 'rel_CREBP', 'rel_colok']]
                       .agg(['mean', 'std', 'sem']).reset_index())

# Flatten multi-level column names
agg_relative_counts.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in agg_relative_counts.columns]

#stats U test
cell_types = ['rel_CREBP_mean', 'rel_cfos_mean', 'rel_colok_mean']

for cell_type in cell_types:
    print(f"\nStatistical Testing for {cell_type}")
    for region in agg_relative_counts['Region'].unique():
        # Extract data for Stim and NoStim groups
        stim_counts = agg_relative_counts.loc[
            (agg_relative_counts['Region'] == region) & (agg_relative_counts['Side'] == 'Stim'), cell_type
        ].dropna()

        nostim_counts = agg_relative_counts.loc[
            (agg_relative_counts['Region'] == region) & (agg_relative_counts['Side'] == 'NoStim'), cell_type
        ].dropna()


        # Skip if groups are empty or have identical values
        if len(stim_counts) == 0 or len(nostim_counts) == 0:
            print(f"Region: {region} has empty groups, skipping test.")
            continue
        if stim_counts.nunique() == 1 and nostim_counts.nunique() == 1:
            print(f"Region: {region} has identical values in both groups, test not applicable.")
            continue

        # Perform the Mann-Whitney U test
        u_stat, p_value = mannwhitneyu(stim_counts, nostim_counts, alternative='two-sided')
        print(f"Region: {region}, U-statistic = {u_stat}, p-value = {p_value}")

#Plot
plt.figure(figsize=(10, 6))

# Create the boxplot using the mean values
sns.boxplot(x='Region', y='rel_cfos_mean', hue='Side', order= ['CTX', 'ACA', 'PL', 'ILA', 'MOp'], palette= ['dimgrey','darkorange'], boxprops=dict(alpha=0.8)
            ,data=agg_relative_counts)

# Get the current axis
ax = plt.gca()

# Get the x-tick positions for the regions
xticks = np.array(ax.get_xticks())

# Calculate offsets for each hue
n_hue = len(agg_relative_counts['Side'].unique())
width = 0.8  # Width of the entire group of boxes
hue_offsets = np.linspace(-width / 3.9, width / 3.9, n_hue)

# Add the error bars
for region_idx, region in enumerate(['CTX', 'ACA', 'PL', 'ILA', 'MOp']):
    for side_idx, side in enumerate(agg_relative_counts['Side'].unique()):
        # Get the subset data for this region and side
        subset = agg_relative_counts[
            (agg_relative_counts['Region'] == region) & (agg_relative_counts['Side'] == side)
        ]
        if not subset.empty:
            mean_val = np.mean(subset['rel_cfos_mean'])
            std_val = np.sqrt(np.sum(subset['rel_cfos_sem']**2))

            # Calculate the x position of the box based on hue offset
            x_pos = xticks[region_idx] + hue_offsets[side_idx]

            # Add the error bar
            plt.errorbar(
                x=x_pos, y=mean_val, yerr=std_val, fmt='o', color='black',
                capsize=5, label='' if region_idx > 0 else side  # Avoid duplicate legend entries
            )
# Customize the plot
plt.title('Boxplot of rel_cfos by Region and averaged per mouse')
plt.xlabel('Region')
plt.ylabel('rel_cfos')
plt.legend(title='Side', loc='upper right')
sns.despine()
plt.savefig(path+'/boxplot_cfos_avg.png')

plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x='Region', y='rel_CREBP_mean', hue='Side', order= ['CTX', 'ACA', 'PL', 'ILA', 'MOp'], palette= ['dimgrey','darkorange'], boxprops=dict(alpha=0.8)
            ,data=agg_relative_counts)


# Get the current axis
ax = plt.gca()

# Get the x-tick positions for the regions
xticks = np.array(ax.get_xticks())

# Calculate offsets for each hue
n_hue = len(agg_relative_counts['Side'].unique())
width = 0.8  # Width of the entire group of boxes
hue_offsets = np.linspace(-width / 3.9, width / 3.9, n_hue)

# Add the error bars
for region_idx, region in enumerate(['CTX', 'ACA', 'PL', 'ILA', 'MOp']):
    for side_idx, side in enumerate(agg_relative_counts['Side'].unique()):
        # Get the subset data for this region and side
        subset = agg_relative_counts[
            (agg_relative_counts['Region'] == region) & (agg_relative_counts['Side'] == side)
        ]
        if not subset.empty:
            mean_val = np.mean(subset['rel_CREBP_mean'])
            std_val = np.sqrt(np.sum(subset['rel_CREBP_sem']**2))

            # Calculate the x position of the box based on hue offset
            x_pos = xticks[region_idx] + hue_offsets[side_idx]

            # Add the error bar
            plt.errorbar(
                x=x_pos, y=mean_val, yerr=std_val, fmt='o', color='black',
                capsize=5, label='' if region_idx > 0 else side  # Avoid duplicate legend entries
            )
plt.title('Boxplot of rel_CREBP by Region and averaged per mouse')
plt.xlabel('Region')
plt.ylabel('rel_CREBP')
plt.legend(title='Side', loc='upper right')
sns.despine()
plt.savefig(path_detect+'/boxplot_CREBP_avg.png')

plt.figure()
# Create the boxplot using the mean values
sns.boxplot(x='Region', y='rel_colok_mean', hue='Side', order= ['CTX', 'ACA', 'PL', 'ILA', 'MOp'], palette= ['dimgrey','darkorange'], boxprops=dict(alpha=0.8)
            ,data=agg_relative_counts)

# Get the current axis
ax = plt.gca()

# Get the x-tick positions for the regions
xticks = np.array(ax.get_xticks())

# Calculate offsets for each hue
n_hue = len(agg_relative_counts['Side'].unique())
width = 0.8  # Width of the entire group of boxes
hue_offsets = np.linspace(-width / 3.9, width / 3.9, n_hue)
plt.title('Boxplot of rel_colok by Region and averaged per mouse')
plt.xlabel('Region')
plt.show()