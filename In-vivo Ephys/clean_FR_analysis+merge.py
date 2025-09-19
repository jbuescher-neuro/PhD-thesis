# -*- coding: utf-8 -*-
"""
Get Firing Frequencies only needs to be run once
Created on Mon Sep 15 12:34:06 2025

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

#%%
#Load all fidelities and get clusters, match unit IDs and exclude ones with bad vorrelation
fid_1= pd.read_csv(exp_path1/ 'FidTest1_fidelity_filtered_with_latency_variance.csv')
fid_1.NeuronID= [x-1 for x in fid_1.NeuronID]

fid_2= pd.read_csv(exp_path1/ 'FidTest2_fidelity_filtered_with_latency_variance.csv')
fid_2.NeuronID= [x-1 for x in fid_2.NeuronID]
#collect good clusters (Fid Test1 is empty)
clus_good1=list(fid_2.NeuronID)

fid_21= pd.read_csv(exp_path2/ 'FidTest1_fidelity_filtered_with_latency_variance.csv')
fid_1.NeuronID= [x-1 for x in fid_1.NeuronID]

fid_22= pd.read_csv(exp_path2/ 'FidTest2_fidelity_filtered_with_latency_variance.csv')
fid_2.NeuronID= [x-1 for x in fid_2.NeuronID]
#collect good clusters (Fid Test1 is empty)
clus_good2=list(fid_22.NeuronID)

cluster_mapping_df1 = pd.read_csv(exp_path1/'phy_KS4_curated2/cluster_si_unit_ids.tsv', sep='\t')
cluster_mapping_df2 = pd.read_csv(exp_path2/'phy_KS4_curated2/cluster_si_unit_ids.tsv', sep='\t')
good_units1= cluster_mapping_df1[cluster_mapping_df1.cluster_id.isin(clus_good1)]
good_units2= cluster_mapping_df2[cluster_mapping_df2.cluster_id.isin(clus_good2)]


#%% Get baseline firing rate and compare first vs second half  if already done skip and load template directly
#get sample range of baseline
#path_to_base= base_folder1 /'experiment4/recording1'
path_to_base= base_folder2 /r'experiment1/recording1'

spike_times=pd.read_csv(exp_path2/'curated_clusters_spike_times_columns_sorted1.csv')
template= pd.read_csv(exp_path2/'curated_template_metrics4.csv', index_col= [0])
quality= pd.read_csv(exp_path2/'curated_quality_metrics.csv', index_col= [0])

quality['si_unit_id']= quality.index



#extract light responsive neurons and non responsive
template['light_response']= 'unresponsive'
template.light_response.loc[good_units2.si_unit_id]= 'responsive'

template['si_unit_id']= template.index
template= template.merge(quality[['firing_rate','firing_range', 'si_unit_id']], how= 'outer', on='si_unit_id')

template=  template.merge(cluster_mapping_df2[['si_unit_id', 'cluster_id']], 
                          how='outer', on='si_unit_id')

base_sample=  np.load(path_to_base/ r'continuous/Acquisition_Board-100.Rhythm Data/sample_numbers.npy')
base_start= 0 #because spike_times starts at 0
base_len=len(base_sample)

#template is ordered by si_unit_id spike times by clusterid!

#split firing rate in half
spikes_base1= ((spike_times > base_start) & (spike_times < base_len*0.5+base_start)).sum(axis=0)
base_dur1= (base_len*0.5)/30000 #(half of base duation)
base1_FR= (spikes_base1/(base_dur1)).values

spikes_base2= ((spike_times > base_start+base_len*0.5) & (spike_times < base_len+base_start)).sum(axis=0) #start at middle oes to end
base_dur2= (base_len*0.5)/30000 #(half of base duation)
base2_FR= (spikes_base2/(base_dur2)).values

# 2. Create a DataFrame with cluster_id from spike_times columns and calculated firing rates
baseline_df = pd.DataFrame({
    'cluster_id': spike_times.columns,  # Assuming each column of spike_times corresponds to a cluster
    'base1_FR': base1_FR,
    'base2_FR': base2_FR
})

# 3. Merge this DataFrame with your template DataFrame on 'cluster_id'
template['cluster_id'] = template['cluster_id'].astype(str)  # or 'object'
baseline_df['cluster_id'] = baseline_df['cluster_id'].astype(str)  # or 'object'
template = pd.merge(template, baseline_df, on='cluster_id', how='left')

template['base_delta']= template.base2_FR-template.base1_FR
print(max(template.base_delta))

# To calculate standard deviation, we need to segment the baseline data into smaller windows.
# For example, let's say we want to calculate std over segments of 1 second (30000 samples).
sampling_rate= 30000
segment_length = 30*sampling_rate  # Number of samples in 30 seconds
num_segments = base_len // segment_length

# Initialize a DataFrame to hold firing rates for each unit and segment
firing_rates_df = pd.DataFrame(index=range(num_segments), columns=spike_times.columns)

# Loop through each segment to calculate firing rates for all units
for i in range(num_segments):
    start_index = i * segment_length
    end_index = start_index + segment_length
    
    # Count spikes in the current segment for all units
    spikes_in_segment =  ((spike_times > start_index) & (spike_times < end_index)).sum()
    
    # Calculate firing rates for this segment for all units
    firing_rate_segment = spikes_in_segment / (segment_length / sampling_rate)  # Duration in seconds
    
    # Store the firing rates in the DataFrame
    firing_rates_df.iloc[i] = firing_rate_segment

# Transpose firing_rates_df so that cluster_id becomes the index
firing_rates_df_transposed = firing_rates_df.T  # Transpose so cluster_id becomes index
firing_rates_df_transposed['base_FR']=firing_rates_df_transposed.mean(axis=1)
firing_rates_df_transposed['base_std']=firing_rates_df_transposed.std(axis=1)
firing_rates_df_transposed['cluster_id']=firing_rates_df_transposed.index


template = template.merge(firing_rates_df_transposed[['base_FR', 'base_std', 'cluster_id']], how='left', on='cluster_id')

#%% other exp
"""for the other experiment but variables the same! 
Doesnot need to be redone !
Get baseline firing rate and compare first vs second half  
if already done skip and load template directly"""
#get sample range of baseline
#path_to_base= base_folder2 /'experiment1/recording1'
path_to_base= base_folder1 /r'experiment4/recording1'

spike_times=pd.read_csv(exp_path1/'curated_clusters_spike_times_columns_sorted1.csv')
template= pd.read_csv(exp_path1/'curated_template_metrics4.csv', index_col= [0])
quality= pd.read_csv(exp_path1/'curated_quality_metrics.csv', index_col= [0])

quality['si_unit_id']= quality.index


#extract light responsive neurons and non responsive
template['light_response']= 'unresponsive'
template.light_response.loc[good_units1.si_unit_id]= 'responsive'

template['si_unit_id']= template.index
template= template.merge(quality[['firing_rate','firing_range', 'si_unit_id']], how= 'outer', on='si_unit_id')

template=  template.merge(cluster_mapping_df1[['si_unit_id', 'cluster_id']], 
                          how='outer', on='si_unit_id')

base_sample=  np.load(path_to_base/ 'continuous/Acquisition_Board-114.Rhythm Data/sample_numbers.npy')
base_start= 0 #because spike_times starts at 0
base_len=len(base_sample)

#template is ordered by si_unit_id spike times by clusterid!

#split firing rate in half
spikes_base1= ((spike_times > base_start) & (spike_times < base_len*0.5+base_start)).sum(axis=0)
base_dur1= (base_len*0.5)/30000 #(half of base duation)
base1_FR= (spikes_base1/(base_dur1)).values

spikes_base2= ((spike_times > base_start+base_len*0.5) & (spike_times < base_len+base_start)).sum(axis=0) #start at middle oes to end
base_dur2= (base_len*0.5)/30000 #(half of base duation)
base2_FR= (spikes_base2/(base_dur2)).values

# 2. Create a DataFrame with cluster_id from spike_times columns and calculated firing rates
baseline_df = pd.DataFrame({
    'cluster_id': spike_times.columns,  # Assuming each column of spike_times corresponds to a cluster
    'base1_FR': base1_FR,
    'base2_FR': base2_FR
})

# 3. Merge this DataFrame with your template DataFrame on 'cluster_id'
template['cluster_id'] = template['cluster_id'].astype(str)  # or 'object'
baseline_df['cluster_id'] = baseline_df['cluster_id'].astype(str)  # or 'object'
template = pd.merge(template, baseline_df, on='cluster_id', how='left')

template['base_delta']= template.base2_FR-template.base1_FR
print(max(template.base_delta))

# To calculate standard deviation, we need to segment the baseline data into smaller windows.
# For example, let's say we want to calculate std over segments of 1 second (30000 samples).
sampling_rate= 30000
segment_length = 30*sampling_rate  # Number of samples in 30 seconds
num_segments = base_len // segment_length

# Initialize a DataFrame to hold firing rates for each unit and segment
firing_rates_df = pd.DataFrame(index=range(num_segments), columns=spike_times.columns)

# Loop through each segment to calculate firing rates for all units
for i in range(num_segments):
    start_index = i * segment_length
    end_index = start_index + segment_length
    
    # Count spikes in the current segment for all units
    spikes_in_segment =  ((spike_times > start_index) & (spike_times < end_index)).sum()
    
    # Calculate firing rates for this segment for all units
    firing_rate_segment = spikes_in_segment / (segment_length / sampling_rate)  # Duration in seconds
    
    # Store the firing rates in the DataFrame
    firing_rates_df.iloc[i] = firing_rate_segment

# Transpose firing_rates_df so that cluster_id becomes the index
firing_rates_df_transposed = firing_rates_df.T  # Transpose so cluster_id becomes index
firing_rates_df_transposed['base_FR']=firing_rates_df_transposed.mean(axis=1)
firing_rates_df_transposed['base_std']=firing_rates_df_transposed.std(axis=1)
firing_rates_df_transposed['cluster_id']=firing_rates_df_transposed.index


template = template.merge(firing_rates_df_transposed[['base_FR', 'base_std', 'cluster_id']], how='left', on='cluster_id')

#%% get firing indiv frequencies for baselines, concatenating missing FOR MORE THAN ONE POST PRIM, Skip if already done
sampling_rate= 30000
"""
path_base_Fid1= base_folder2 / r'experiment1/recording3'
path_base_prim1= base_folder2/ r"experiment1/recording5"
path_base_prim2= base_folder2 / r"experiment1/recording6"
path_base_prim3= base_folder2 / r"experiment1/recording7"
path_base_Fid2 = base_folder2 / "experiment1/recording9"

#Acqu Board1
fid1= np.load(base_folder2 / 'experiment1/recording2/continuous/Acquisition_Board-100.Rhythm Data/sample_numbers.npy')
prim= np.load(base_folder2 / 'experiment1/recording4/continuous/Acquisition_Board-100.Rhythm Data/sample_numbers.npy')
fid2= np.load(base_folder2 / 'experiment1/recording8/continuous/Acquisition_Board-100.Rhythm Data/sample_numbers.npy')

base_Fid1_sample=  np.load(path_base_Fid1/ 'continuous/Acquisition_Board-100.Rhythm Data/sample_numbers.npy')
base_Prim1_sample=  np.load(path_base_prim1/ 'continuous/Acquisition_Board-100.Rhythm Data/sample_numbers.npy')
base_Prim2_sample=  np.load(path_base_prim2/ 'continuous/Acquisition_Board-100.Rhythm Data/sample_numbers.npy')
base_Prim3_sample=  np.load(path_base_prim3/ 'continuous/Acquisition_Board-100.Rhythm Data/sample_numbers.npy')
base_Fid2_sample=  np.load(path_base_Fid2/ 'continuous/Acquisition_Board-100.Rhythm Data/sample_numbers.npy')
"""

path_base_Fid1= base_folder1 / r'experiment4/recording3'
path_base_prim1= base_folder1 / r"experiment4/recording5"
path_base_prim2= base_folder1 / r"experiment4/recording6"
path_base_prim3= base_folder1 / r"experiment4/recording7"
path_base_Fid2 = base_folder1 / "experiment4/recording9"

fid1= np.load(base_folder1 / 'experiment4/recording2/continuous/Acquisition_Board-114.Rhythm Data/sample_numbers.npy')
prim= np.load(base_folder1 / 'experiment4/recording4/continuous/Acquisition_Board-114.Rhythm Data/sample_numbers.npy')
fid2= np.load(base_folder1 / 'experiment4/recording8/continuous/Acquisition_Board-114.Rhythm Data/sample_numbers.npy')

base_Fid1_sample=  np.load(path_base_Fid1/ 'continuous/Acquisition_Board-114.Rhythm Data/sample_numbers.npy')
base_Prim1_sample=  np.load(path_base_prim1/ 'continuous/Acquisition_Board-114.Rhythm Data/sample_numbers.npy')
base_Prim2_sample=  np.load(path_base_prim2/ 'continuous/Acquisition_Board-114.Rhythm Data/sample_numbers.npy')
base_Prim3_sample=  np.load(path_base_prim3/ 'continuous/Acquisition_Board-114.Rhythm Data/sample_numbers.npy')
base_Fid2_sample=  np.load(path_base_Fid2/ 'continuous/Acquisition_Board-114.Rhythm Data/sample_numbers.npy')


#Account for not recorded samples!
start_base = 0
end_base = len(base_sample) 
#base_len end of first recording
start_fid1= base_len+1
end_rec2= len(fid1)+ base_len+1 # end of fidTest1 = rec2
start_base_fid1= end_rec2+1
end_base_fid1= len(base_Fid1_sample)+ start_base_fid1

start_prim= end_base_fid1+1
end_prim= end_base_fid1+1+len(prim)
start_base_prim= end_prim+1
end_base_prim= len(base_Prim1_sample)+ start_base_prim

start_base_prim2= end_base_prim+1
end_base_prim2= end_base_prim+1+ len(base_Prim2_sample)
start_base_prim3= end_base_prim2+1
end_base_prim3=end_base_prim2+ 1+ len(base_Prim3_sample)

start_fid2= end_base_prim3+1
end_fid2= end_base_prim3+1+ len(fid2)
start_base_fid2= end_fid2+1
end_base_fid2=start_base_fid2+ len(base_Fid2_sample)

print(f'End Fidtest1:{end_rec2/30000/60}, Start and end fid1_base: {start_base_fid1/30000/60}, {end_base_fid1/30000/60},'
      ,'End priming {end_prim/30000/60}, start and end base_prim: {start_base_prim/30000/60}, {end_Base_prim/30000/60}',
      'End base_fid2: {start_base_fid2/30000/60}, {end_base_fid2/30000/60}')
#order template to cluster ids for easier take over
template_sorted = template.sort_values(by='cluster_id').reset_index(drop=True)

spikes_base_Fid1= ((spike_times > start_base_fid1) & (spike_times < end_base_fid1)).sum()
base_dur1= (len(base_Fid1_sample))/30000 

template_sorted['FR_postFid1']= (spikes_base_Fid1/(base_dur1)).values

spikes_base_Fid2= ((spike_times > start_base_fid2) & (spike_times < end_base_fid2)).sum()
base_dur1= (len(base_Fid2_sample))/30000 

template_sorted['FR_postFid2']= (spikes_base_Fid2/(base_dur1)).values

spikes_Fid1= ((spike_times > start_fid1) & (spike_times < end_rec2)).sum()
dur= (len(fid1))/30000 
template_sorted['FR_Fid1']= (spikes_Fid1/(dur)).values

spikes_Fid2= ((spike_times > start_fid2) & (spike_times < end_fid2)).sum()
dur= (len(fid2))/30000 
template_sorted['FR_Fid2']= (spikes_Fid2/(dur)).values

spikes_Prim= ((spike_times > start_prim) & (spike_times < end_prim)).sum()
dur= (len(prim))/30000 
template_sorted['FR_Prim']= (spikes_Prim/(dur)).values


#getting the firing frequency for every 5 minutes in the post prim baseline
window= 5*60 #5 min window
segment_length = window*sampling_rate  # Number of samples in in window (5 min)
num_segments =  (end_base_prim3 - start_base_prim) / segment_length
num_segments = int(np.ceil((end_base_prim3 - start_base_prim) / segment_length))

# Initialize a DataFrame to hold firing rates for each unit and segment
firing_rates_df = pd.DataFrame(index=range(num_segments), columns=spike_times.columns)

# Loop through each segment to calculate firing rates for all units
prim_start_sample = start_base_prim
for i in range(num_segments):
    start_sample = prim_start_sample + i * segment_length
    end_sample = start_sample + segment_length
    
    # Count spikes in the current segment for all units
    spikes_in_segment =  ((spike_times > start_sample) & (spike_times < end_sample)).sum()
    
    # Calculate firing rates for this segment for all units
    firing_rate_segment = spikes_in_segment / (segment_length / sampling_rate)  # Duration in seconds
    
    # Store the firing rates in the DataFrame
    firing_rates_df.iloc[i] = firing_rate_segment

# Now calculate mean and std for each unit's firing rates across segments
mean_firing_rates = firing_rates_df.mean()
std_firing_rates = firing_rates_df.std()


# Merge responsiveness info
unit_meta = template[['cluster_id', 'light_response']]

firing_rates_labeled = firing_rates_df.T  # transpose: now rows = units
firing_rates_labeled['cluster_id'] = firing_rates_labeled.index
firing_rates_labeled['cluster_id'] = firing_rates_labeled['cluster_id'].astype(int)
unit_meta['cluster_id'] = unit_meta['cluster_id'].astype(int)
firing_rates_labeled = firing_rates_labeled.merge(unit_meta, on='cluster_id')


#merge first and last cluster to template
if 'FR_postPrim1' not in template.columns and 'FR_postPrim'+str(num_segments) not in template.columns:
    # Select relevant columns from firing_rates_labeled
    print(num_segments)
    postPrim_subset = firing_rates_labeled[['cluster_id', 0, 3, firing_rates_labeled.columns[-3]]]
    
    postPrim_subset = postPrim_subset.rename(columns={0: 'FR_postPrim1', 3: 'FR_postPrim4',
                                                       firing_rates_labeled.columns[-3]: 'FR_postPrim'+str(num_segments)})
    
    # Merge into template
    template_sorted['cluster_id'] = template_sorted['cluster_id'].astype(int)
    template_sorted = template_sorted.merge(postPrim_subset, on='cluster_id', how='left')
# Calculate the Z-score for FR_postPrim4 compared to baseline
epsilon = 1e-5  # Small constant to avoid division by zero
template_sorted['Z_FR_postPrim4'] = (template_sorted['FR_postPrim4'] - template_sorted['base_FR']) / (template_sorted['base_std']+ epsilon)

template=template_sorted

template['FR_delta']= template['FR_postPrim4']-template['base_FR']
template['FR_delta_fid']= template['FR_postFid2']- template['FR_postFid1']

template['delta_postResTest1']= template.FR_postFid1 -template.base_FR
template['delta_postResTest2']= template.FR_postFid2 -template.base_FR
template['delta_postPrim1']= template.FR_postPrim1 -template.base_FR
template['delta_postPrim4']= template.FR_postPrim4 -template.base_FR
template['delta_postPrim7']= template.FR_postPrim7 -template.base_FR

template.to_csv(exp_path1/'template_analysis1.csv')
#print('template saved to ', exp_path)

#%%

#%% merge template from different experiments
combined_path= Path('E:/PhD/2. In-vivo Ephys Data/result/')


template1=pd.read_csv(exp_path1/'template_analysis1.csv')
template2=pd.read_csv(exp_path2/'template_analysis1.csv')

template1['si_unit_id']= 'B6J-8464_'+template1['si_unit_id'].astype(str)
template2['si_unit_id']= 'B6J-8636_'+template2['si_unit_id'].astype(str)

combined_template= pd.concat([template1,template2], ignore_index=True)
template=combined_template

template.to_csv(combined_path/'template1.csv')

#Same for Fideilty
combined_fid= pd.concat([fid_2,fid_22], ignore_index=True)

combined_fid.to_csv(combined_path/'fidelity1_fidtest2.csv')

#%%Merge Latencies of all units

template1=pd.read_csv(exp_path1/r'template_analysis1.csv')
template2=pd.read_csv(exp_path2/r'template_analysis1.csv')

#Second light responsiveness Test
fid2_1= pd.read_csv(exp_path1/ 'recording8_latencies.csv')#fidelity_filtered_with_latency_columns.csv')
fid2_1.NeuronID= [x-1 for x in fid2_1.NeuronID]
fid2_1['Source']= 'B6J-8646'
# Merge: match NeuronID in fid2_1 with cluster_id in template1
fid2_1 = fid2_1.merge(
    template1[['cluster_id', 'light_response']],  # only need these columns
    left_on='NeuronID',
    right_on='cluster_id',
    how='left'   # keeps all rows in fid2_1
)
# Drop duplicate cluster_id 
fid2_1 = fid2_1.drop(columns=['cluster_id'])

fid2_2= pd.read_csv(exp_path2/ 'recording8_latencies.csv')
fid2_2.NeuronID= [x-1 for x in fid2_2.NeuronID]
fid2_2['Source']= 'B6J-8636'
# Merge: match NeuronID in fid2_1 with cluster_id in template1
fid2_2 = fid2_2.merge(
    template2[['cluster_id', 'light_response']],  # only need these columns
    left_on='NeuronID',
    right_on='cluster_id',
    how='left'   # keeps all rows in fid2_1
)

# Drop duplicate cluster_id if you don’t want it twice
fid2_2 = fid2_2.drop(columns=['cluster_id'])

combined_lat= pd.concat([fid2_1,fid2_2], ignore_index=True)
combined_lat["Neuron_ID"] = combined_lat["Source"].astype(str) + "_" + combined_lat["NeuronID"].astype(str)

combined_lat.to_csv(combined_path/'lats_FidTest2.csv')

#First Light responsiveness test

fid2_1= pd.read_csv(exp_path1/ 'recording2_latencies.csv')#fidelity_filtered_with_latency_columns.csv')
fid2_1.NeuronID= [x-1 for x in fid2_1.NeuronID]
fid2_1['Source']= 'B6J-8646'
# Merge: match NeuronID in fid2_1 with cluster_id in template1
fid2_1 = fid2_1.merge(
    template1[['cluster_id', 'light_response']],  # only need these columns
    left_on='NeuronID',
    right_on='cluster_id',
    how='left'   # keeps all rows in fid2_1
)
# Drop duplicate cluster_id if you don’t want it twice
fid2_1 = fid2_1.drop(columns=['cluster_id'])

fid2_2= pd.read_csv(exp_path2/ 'recording2_latencies.csv')
fid2_2.NeuronID= [x-1 for x in fid2_2.NeuronID]
fid2_2['Source']= 'B6J-8636'
# Merge: match NeuronID in fid2_1 with cluster_id in template1
fid2_2 = fid2_2.merge(
    template2[['cluster_id', 'light_response']],  # only need these columns
    left_on='NeuronID',
    right_on='cluster_id',
    how='left'   # keeps all rows in fid2_1
)
# Drop duplicate cluster_id if you don’t want it twice
fid2_2 = fid2_2.drop(columns=['cluster_id'])

combined_lat2= pd.concat([fid2_1,fid2_2], ignore_index=True)
combined_lat2["Neuron_ID"] = combined_lat2["Source"].astype(str) + "_" + combined_lat2["NeuronID"].astype(str)

combined_lat2.to_csv(combined_path/'lats_FidTest1.csv')
