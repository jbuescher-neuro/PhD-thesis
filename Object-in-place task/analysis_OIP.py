# -*- coding: utf-8 -*-
"""
Used functions from clean_analysis
Created on Wed May  7 10:43:45 2025

@author: jbuesche
"""
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import pingouin as pg
from sklearn.linear_model import LinearRegression
#from statsmodels.formula.api import mixedlm
#import scikit_posthocs as sp
from scipy.stats import ttest_ind, mannwhitneyu, kruskal, chi2_contingency, ttest_rel
from scipy.stats import shapiro, levene
from scipy.stats import friedmanchisquare, f_oneway, wilcoxon
from statsmodels.stats.anova import AnovaRM


#%% Creating one Dataframe with data from boris file and experiment information
def boris2analysis (path, comp):
    """takes path to folder with Boris Data from all experiments and gives same data as automated analysis. Make for every scorer own df?"""
    object_pos= pd.read_excel(path+'object_pos_group.xlsx', index_col=0)
    object_pos.index=object_pos.index.astype(str)
    fps=30
    csv_count = sum(1 for file in os.listdir(path+'results') if file.endswith('.csv'))
    print(csv_count)
    visits= pd.DataFrame()
    df= pd.DataFrame(index=np.arange(csv_count+1), columns= ['MouseID','Task','Object1_min1', 'Object2_min1', 'Object3_min1', 'Object4_min1', 'Object1_min2', 'Object2_min2', 'Object3_min2', 'Object4_min2',
                                 'Object1_min3', 'Object2_min3', 'Object3_min3', 'Object4_min3', 'Object1_min4', 'Object2_min4', 'Object3_min4', 'Object4_min4',
                                 'Object1_min5', 'Object2_min5', 'Object3_min5', 'Object4_min5', 'Object1_min6', 'Object2_min6', 'Object3_min6', 'Object4_min6',
                                 'Visits_Object1', 'Visits_Object2', 'Visits_Object3','Visits_Object4',
                                 'group'])
                       
    for idx, filename in enumerate(os.listdir(path+'results')):
        
        if filename.endswith('.csv'):
            print(idx, filename)
            df_min= df.iloc[idx]
            boris= pd.read_csv(path+'results/'+filename, usecols= ['Behavior', 'Start (s)', 'Stop (s)', 'Duration (s)'], header= 0)
            
            
            vid_name= filename.split('.')[0]
            exp= vid_name.split('_')[-1]
            
            """for scoring videos"""
            if comp == True:
                exp= vid_name.split('_')[-1]   
                vid_name= vid_name.split('_')[0]
           
            
            #get Object number for each location and put it in boris file create column in df file
            boris['object']= 0 
            filename= filename.split('.')[0]    
            for pos in object_pos.columns[0:4]:
                
                boris.loc[boris.Behavior==pos, 'object']= 'Object'+ str(int(object_pos[pos].loc[vid_name]))
            #print(object_pos.Animal_ID[object_pos.index==vid_name][0])
            df_min['MouseID']= object_pos.Animal_ID[object_pos.index==vid_name][0]
            df_min['Task']= object_pos.Test[object_pos.index==vid_name][0]
            df_min['group']= object_pos.group[object_pos.index==vid_name][0]   
            
    #get the exploration frames per object
            frames1= pd.Series(name= 'Object1',dtype= int)
            frames2= pd.Series(name= 'Object2',dtype= int)
            frames3= pd.Series(name= 'Object3',dtype= int)
            frames4= pd.Series(name= 'Object4',dtype= int)
            for idx, row in boris.iterrows():
                
                if boris['object'].iloc[idx]== 'Object1':
                    frames1=pd.concat([frames1, pd.Series(range(int(np.round((row['Start (s)'])*fps)), int(np.round(row['Stop (s)']*fps+1))))], axis=0, ignore_index=True)
                elif boris['object'].iloc[idx]== 'Object2':
                    frames2=pd.concat([frames2, pd.Series(range(int(np.round((row['Start (s)'])*fps)), int(np.round(row['Stop (s)']*fps+1))))], axis=0, ignore_index=True)
                elif boris['object'].iloc[idx]== 'Object3':
                    frames3=pd.concat([frames3, pd.Series(range(int(np.round((row['Start (s)'])*fps)), int(np.round(row['Stop (s)']*fps+1))))], axis=0, ignore_index=True)
                elif boris['object'].iloc[idx]== 'Object4':
                    frames4=pd.concat([frames4, pd.Series(range(int(np.round((row['Start (s)'])*fps)), int(np.round(row['Stop (s)']*fps+1))))], axis=0, ignore_index=True)
            expl_df= pd.DataFrame({'Object1':frames1, 'Object2':frames2, 'Object3':frames3, 'Object4':frames4})
            
            #expl_df.to_csv(os.path.dirname(os.path.dirname(path))+'/OpenCV/Expl/expl_frames_'+str(object_pos.Filename[object_pos.index==vid_name][0])+'.csv', index= False)
            
    #get obj_exploration per minute
            for minu in range(1,7):
                
                df_min.loc[['Object1_min'+str(minu),'Object2_min'+str(minu),'Object3_min'+str(minu),'Object4_min'+str(minu)]]=pd.Series([expl_df.Object1[expl_df.Object1<fps*minu*60].count()/fps, expl_df.Object2[expl_df.Object2<fps*minu*60].count()/fps, expl_df.Object3[expl_df.Object3<fps*minu*60].count()/fps, expl_df.Object4[expl_df.Object4<fps*minu*60].count()/fps], 
                                                                                                                                                                                                                                                 index=['Object1_min'+str(minu),'Object2_min'+str(minu),'Object3_min'+str(minu),'Object4_min'+str(minu)])
            
            #get scorer, total exploration per object, visits per object and expl ratio
            #df_min.loc['scorer',object_pos.index[object_pos.index==vid_name][0]]= exp
            
            
            
            df_min.loc[['Visits_Object1', 'Visits_Object2', 'Visits_Object3','Visits_Object4']]= pd.Series([boris['Duration (s)'][boris.object=='Object1'].count(), boris['Duration (s)'][boris.object=='Object2'].count(),
                                     boris['Duration (s)'][boris.object=='Object3'].count(), boris['Duration (s)'][boris.object=='Object4'].count()],
                                    index= ['Visits_Object1', 'Visits_Object2', 'Visits_Object3','Visits_Object4'])
    
    
    print(df)
    for idx,i in enumerate(range(2,26,4)):
        df= df.dropna()
        df['disc_ratio_min'+str(idx+1)]= df.apply(lambda row: ((row['Object3_min'+str(idx+1)]+row['Object4_min'+str(idx+1)])-(row['Object1_min'+str(idx+1)]+row['Object2_min'+str(idx+1)]))/(row['Object1_min'+str(idx+1)]+row['Object2_min'+str(idx+1)]+row['Object3_min'+str(idx+1)]+row['Object4_min'+str(idx+1)]),axis=1)
        
    df=df.dropna()
    print(df)
    df.to_csv(path+'df_min.csv')
    return(boris, expl_df, df)
#%% not so important single functions

def groups(path, minute, test):
    df= pd.read_csv(path+'df_min.csv', index_col=0, header=0)
    #df_min=df_min.reindex(sorted(df_min.columns), axis=1)
 
    #Boxplot
    df['disc_ratio_min'+str(minute)]=df['disc_ratio_min'+str(minute)].astype(float)
    plt.figure(figsize= (4,3.54), dpi= 300)
    ax= plt.subplot()
    sns.boxplot(x='group', y='disc_ratio_min'+str(minute), data=df.loc[df.Task.str.contains(test)], order= ['Control', 'Chrim'], palette= {'Control':'dimgray','Chrim':'darkorange'}, saturation=1, boxprops=dict(alpha=.45), width=0.4)
    sns.stripplot(x='group', y='disc_ratio_min'+str(minute), data=df.loc[df.Task.str.contains(test)], hue='group', hue_order=['Control','Chrim'], order= ['Control', 'Chrim'], palette= {'Control':'dimgray','Chrim':'darkorange'}, size=7.5, edgecolor='k', linewidth=1, legend= False)
    plt.title('Disc ratio minute '+str(minute)+ ' '+ test)
    stat, p= stats.ttest_ind(df['disc_ratio_min'+str(minute)][(df.Task==test) & (df.group=='Chrim')], df['disc_ratio_min'+str(minute)][(df.Task==test) & (df.group=='Control')], alternative='greater')
    #plt.text(-0.8,-0.7,'Ttest p value= ' +str(p))
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylabel('disc ratio min '+str(minute))
    plt.tight_layout()
    #plt.axhline(0, 0, color= 'k')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    #plt.grid()
    plt.savefig(path+test+'ChrimvsControl'+str(minute)+'.svg')
    plt.savefig(path+test+'ChrimvsControl'+str(minute)+'.png')
    print('Ttest bewteen groups p value= ' +str(p))
    
    #test if group are different from 0 -> learned
    stat, p_chrim= stats.ttest_1samp(df['disc_ratio_min'+str(minute)][(df.Task==test) & (df.group=='Chrim')], 0)
    stat, p_con= stats.ttest_1samp(df['disc_ratio_min'+str(minute)][(df.Task==test) & (df.group=='Control')], 0)
    print(f'Chrimson to 0 p={p_chrim}, Control to 0 p={p_con}')
    
    
    #Disc_ratio per minute over test for every mouse
    df_disc=df.loc[:,~df.columns.str.contains('Object')]
    df_melt= df_disc.melt(['MouseID','Task','group'], var_name='minute', value_name='disc_ratio')
    
    plt.figure(figsize= (4,3.54), dpi= 300)
    ax= plt.subplot()
    #get number of Chrim and control animals for coloring
    num_chrim= len(df_melt['MouseID'].loc[(df_melt.group=='Chrim') & (df_melt.Task==test)].unique())
    num_con= len(df_melt['MouseID'].loc[(df_melt.group=='Control') & (df_melt.Task==test)].unique())
   
    sns.lineplot(data= df_melt.loc[(df_melt.Task==test) ], x='minute', y='disc_ratio', hue='group', hue_order=['Control', 'Chrim'], palette= ['dimgrey','darkorange'], legend=False, 
                 errorbar=('se'))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    #to get individual animals
    #sns.lineplot(data= df_melt.loc[(df_melt.Task==test)& (df_melt.group=='Chrim')], x='minute', y='disc_ratio', hue='MouseID', palette=['navajowhite']*num_chrim)
    #sns.lineplot(data= df_melt.loc[(df_melt.Task==test)& (df_melt.group=='Control')], x='minute', y='disc_ratio', hue='MouseID', palette=['gainsboro']*num_con)
    
    plt.xticks([0,1,2,3,4,5], ['1', '2', '3', '4', '5', '6'])
    plt.xlabel('minute')
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylabel('disc ratio')
    plt.title('cumulative disc ratio in '+ test)
    plt.axhline(0, 0, color= 'k')
    #plt.grid()
    plt.tight_layout()
    plt.savefig(path+test+'ChrimvsControl_perMinute'+'.svg')
    plt.savefig(path+test+'ChrimvsControl_perMinute'+'.png')
    
    return(df)
#%%exploration
def expl(path):
    """exploration times per object (separated by group) for every phase (min!)
    - exploration times per object (separated by group) for all test and study phases
    - exploration times in every phase displaced vs non displaced objects (min3!)
    - Per phase and group displaced vs non-displaced objects
    - paired ttest or wilcoxon every group/phase displaced vs non-displaced"""
    df = pd.read_csv(path + 'df_min.csv', index_col=0, header=0)
    df= df.drop(df.loc[df.MouseID=='B6J-7037'].index)
    expl = df.loc[:, ('MouseID', 'Task', 'group', 'Object1_min6', 'Object2_min6', 'Object3_min6', 'Object4_min6')]
    expl = expl.melt(['MouseID', 'Task', 'group'], var_name='Object', value_name='expl_time')
    expl_disp = df.loc[:, ('MouseID', 'Task', 'group', 'Object1_min3', 'Object2_min3', 'Object3_min3', 'Object4_min3')]
    expl_disp = expl_disp.melt(['MouseID', 'Task', 'group'], var_name='Object', value_name='expl_time')
    
    # Classify objects as Non-displaced or Displaced
    expl_disp['Object_type'] = expl_disp['Object'].apply(lambda x: 'Non-displaced' if x in ['Object1_min3', 'Object2_min3'] else 'Displaced')
    
    # Plot object exploration per object for each task separately (Study + Test)
    for task in expl['Task'].unique():
       # Filter for the current task
       task_expl = expl[expl['Task'] == task]
       
       # Plot the data for each task
       plt.figure(figsize=(6, 4), dpi=300)
       ax = plt.subplot()
       sns.boxplot(data=task_expl, x='Object', y='expl_time', hue='group',
                   hue_order=['Control', 'Chrim'], palette={'Control': 'dimgray', 'Chrim': 'darkorange'},
                   saturation=1, boxprops=dict(alpha=.45), width=0.6)
       plt.xticks([0, 1, 2, 3], labels=['Object1', 'Object2', 'Object3', 'Object4'])
       ax.spines['right'].set_visible(False)
       ax.spines['top'].set_visible(False)
       plt.title(f'Object Exploration in {task} (per Object)')
       plt.ylabel('Exploration time [s]')
       plt.xlabel(' ')
       plt.tight_layout()
       plt.savefig(f"{path}{task}_ObjExpl_PerObject.svg")
       print(f'saved in {path}{task}_ObjExpl_PerObject.svg')
       
       for group in expl['group'].unique():
            group_data = task_expl[task_expl['group'] == group]
            group_data_pivot = group_data.pivot(index='MouseID', columns='Object', values='expl_time')
            
            # Check if there are enough data points
            if group_data_pivot.shape[0] > 3:
                # Check for normality for each object (across all mice)
                _, p_shapiro = shapiro(group_data_pivot.values.flatten())
                
                if p_shapiro > 0.05:
                    # Normal data, use One-Way Repeated Measures ANOVA
                    try:
                        group_data_long = group_data_pivot.reset_index().melt(id_vars='MouseID', var_name='Object', value_name='expl_time')

                        anova = AnovaRM(group_data_long, depvar='expl_time', subject='MouseID', within=['Object']).fit()
                        p = anova.anova_table['Pr > F'][0]
                        test_type = "Repeated Measures ANOVA"
                        print(f'{task} between all objects ({group}) - {test_type}: p = {p:.4f}')
                    
                    except Exception:
                        print(f"ANOVA failed for {task} ({group}): ")
                        p = 1.0
                        test_type = "ANOVA failed"
                else:
               
                    # Non-normal data, use Friedman test
                    _, p = friedmanchisquare(*[group_data_pivot[obj].dropna() for obj in group_data_pivot.columns])
                    test_type = "Friedman Test"
                    
                    print(f'{task} between all objects ({group}) - {test_type}: p = {p:.4f}')
                
                # If the test is significant, perform pairwise comparisons
                if p < 0.05:
                    print(f"Pairwise comparisons for {task} ({group}):")
                    for obj1, obj2 in [('Object1_min6', 'Object2_min6'), ('Object1_min6', 'Object3_min6'),
                                       ('Object1_min6', 'Object4_min6'), ('Object2_min6', 'Object3_min6'),
                                       ('Object2_min6', 'Object4_min6'), ('Object3_min6', 'Object4_min6')]:
                        # Extract exploration time data for the two objects
                        obj1_data = group_data_pivot[obj1].dropna()
                        obj2_data = group_data_pivot[obj2].dropna()
                        
                        # Perform Wilcoxon signed-rank test for pairwise comparison
                        stat, p = wilcoxon(obj1_data, obj2_data)
                        print(f'  {obj1} vs {obj2}: p = {p:.4f}')
    # Plot object exploration for displaced and non-displaced objects for all test phases
    summed_expl = expl_disp.groupby(['MouseID', 'Task', 'group', 'Object_type']).agg({'expl_time': 'sum'}).reset_index()
    
    for task in summed_expl.Task.unique():
        # Filter for Test phases
        test_expl = summed_expl[summed_expl['Task'] == task]
        
        plt.figure(figsize=(6, 4), dpi=300)
        ax = plt.subplot()
        
        # Boxplot + Stripplot for displaced vs non-displaced objects in Test phases
        sns.boxplot(data=test_expl, x='Object_type', y='expl_time', hue='group',
                    hue_order=['Control', 'Chrim'], palette={'Control': 'dimgray', 'Chrim': 'darkorange'},
                    saturation=1, boxprops=dict(alpha=.45), width=0.6)
        sns.stripplot(data=test_expl, x='Object_type', y='expl_time', hue='group',
                      hue_order=['Control', 'Chrim'], palette={'Control': 'dimgray', 'Chrim': 'darkorange'},
                      size=5.5, alpha=.8, edgecolor='k', linewidth=1, legend=False, dodge=True, jitter=0.1)
        
        handles, labels = ax.get_legend_handles_labels()
        labels = ['Control' if lbl == 'Control' else 'Primed' for lbl in labels]
        ax.legend(handles=handles[:2], labels=labels[:2], title='Group')  # [:2] to avoid duplicate legend entries
        
        plt.title(f'Object Exploration in {task} (Displaced vs Non-displaced)')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.ylabel('Exploration time [s]')
        plt.tight_layout()
        plt.savefig(f"{path}{task}_ObjExpl_NonDisplaced_vs_Displaced.svg")
        
        for group in test_expl['group'].unique():
                group_data = test_expl[test_expl['group'] == group]

                # Pivot so we get one row per mouse with columns: 'Displaced' and 'Non-displaced'
                pivot = group_data.pivot(index='MouseID', columns='Object_type', values='expl_time')

                if pivot.shape[0] > 3 and all(col in pivot.columns for col in ['Displaced', 'Non-displaced']):
                    # Shapiro-Wilk for normality
                    _, p_shapiro_disp = shapiro(pivot['Displaced'].dropna())
                    _, p_shapiro_nondisp = shapiro(pivot['Non-displaced'].dropna())

                    if p_shapiro_disp > 0.05 and p_shapiro_nondisp > 0.05:
                        stat, p = ttest_rel(pivot['Displaced'], pivot['Non-displaced'])
                        test_type = "Paired t-test"
                    else:
                        stat, p = wilcoxon(pivot['Displaced'], pivot['Non-displaced'])
                        test_type = "Wilcoxon"

                    print(f'{task} ({group}): Displaced vs Non-displaced - {test_type}: p = {p:.4f}')
    """Tests if Chrim and Control are different in displaced or non displaced
        for obj_type in ['Non-displaced', 'Displaced']:
            control_data = test_expl.loc[(test_expl['Object_type'] == obj_type) & (test_expl['group'] == 'Control'), 'expl_time'].dropna()
            chrim_data = test_expl.loc[(test_expl['Object_type'] == obj_type) & (test_expl['group'] == 'Chrim'), 'expl_time'].dropna()
            
            # Perform Shapiro-Wilk Test for normality
            _, p_shapiro_control = shapiro(control_data)
            _, p_shapiro_chrim = shapiro(chrim_data)
            
            # Perform the correct statistical test based on normality
            if p_shapiro_control > 0.05 and p_shapiro_chrim > 0.05:
                stat, p = ttest_ind(control_data, chrim_data)
                test_type = "t-test"
            else:
                stat, p = mannwhitneyu(control_data, chrim_data, alternative='two-sided')
                test_type = "Mann-Whitney U"
                
            print(f'{task} ({obj_type}) - {test_type}: p = {p:.4f}')"""
            
    # Plot object exploration for all study and all test phases together (per object)
    for phase in ['Study', 'Test']:
        phase_expl = expl[expl.Task.str.contains(phase)]
        
        plt.figure(figsize=(6, 4), dpi=300)
        ax = plt.subplot()
        sns.boxplot(data=phase_expl, x='Object', y='expl_time', hue='group',
                    hue_order=['Control', 'Chrim'], palette={'Control': 'dimgray', 'Chrim': 'darkorange'},
                    saturation=1, boxprops=dict(alpha=.45), width=0.6)
        plt.xticks([0, 1, 2, 3], labels=['Object1', 'Object2', 'Object3', 'Object4'])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.title(f'Object Exploration in {phase} Phases (All Objects)')
        plt.ylabel('Exploration time [s]')
        plt.xlabel(' ')
        plt.tight_layout()
        plt.savefig(f"{path}{phase}_ObjExpl_AllObjects.svg")
        """
        # Statistical Tests for total exploration (All Objects) per group
        control_data = phase_expl.loc[phase_expl['group'] == 'Control', 'expl_time'].dropna()
        chrim_data = phase_expl.loc[phase_expl['group'] == 'Chrim', 'expl_time'].dropna()
        
        # Perform Shapiro-Wilk Test for normality
        _, p_shapiro_control = shapiro(control_data)
        _, p_shapiro_chrim = shapiro(chrim_data)
        
        # Perform the correct statistical test based on normality
        if p_shapiro_control > 0.05 and p_shapiro_chrim > 0.05:
            stat, p = ttest_ind(control_data, chrim_data)
            test_type = "t-test"
        else:
            stat, p = mannwhitneyu(control_data, chrim_data, alternative='two-sided')
            test_type = "Mann-Whitney U"
        """        
        #print(f'{phase} Phases (All Objects) - {test_type}: p = {p:.4f}')
        
    # Check total exploration difference for each task
    total_expl_per_task = expl.groupby(['MouseID', 'Task', 'group']).agg({'expl_time': 'sum'}).reset_index()
    plt.figure(figsize=(6, 4), dpi=300)
    ax = plt.subplot()
    
    # Boxplot of total exploration time per group
    sns.boxplot(data=total_expl_per_task, x='Task', y='expl_time', hue='group',
                hue_order=['Control', 'Chrim'], palette={'Control': 'dimgray', 'Chrim': 'darkorange'},
                saturation=1, boxprops=dict(alpha=.45), width=0.6)
    sns.stripplot(data=total_expl_per_task, x='Task', y='expl_time', hue= 'group',
                  palette={'Control': 'dimgray', 'Chrim': 'darkorange'},hue_order=['Control', 'Chrim'],
                  size=5.5, alpha=.8, edgecolor='k', linewidth=1, legend=False, dodge=True, jitter=0.1)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    labels = ['Control' if lbl == 'Control' else 'Primed' for lbl in labels]
    ax.legend(handles=handles[:2], labels=labels[:2], title='Group')
    plt.title('Total Exploration Time Across All Tasks')
    plt.ylabel('Exploration time [s]')
    plt.xlabel('Group')
    plt.tight_layout()
    plt.savefig(f"{path}TotalExploration_AllTasks_PerGroup.svg")
    
    #anova is better practice
    for task in total_expl_per_task.Task.unique():
        task_data = total_expl_per_task[total_expl_per_task['Task'] == task]
        
        control_data = task_data[task_data['group'] == 'Control']['expl_time'].dropna()
        chrim_data = task_data[task_data['group'] == 'Chrim']['expl_time'].dropna()
        
        # Perform Shapiro-Wilk Test for normality
        _, p_shapiro_control = shapiro(control_data)
        _, p_shapiro_chrim = shapiro(chrim_data)
        """
        # Perform the correct statistical test based on normality
        if p_shapiro_control > 0.05 and p_shapiro_chrim > 0.05:
            stat, p = ttest_ind(control_data, chrim_data)
            test_type = "t-test"
        else:
            stat, p = mannwhitneyu(control_data, chrim_data, alternative='two-sided')
            test_type = "Mann-Whitney U"
        
        print(f'{task} - Total Exploration between Groups: {test_type}: p = {p:.4f}')"""
    if p_shapiro_control > 0.05 and p_shapiro_chrim > 0.05:
        anova_data = total_expl_per_task.dropna(subset=['expl_time'])
    # Mixed ANOVA with pingouin
        anova_results = pg.mixed_anova(dv='expl_time', between= 'group', within= 'Task', subject= 'MouseID', data= anova_data)
        #pg.rm_anova(dv='expl_time', within= 'Task', subject= 'MouseID', data= anova_data)
        print("\Mixed ANOVA Results (Total Exploration across Tasks and Groups):")
        print(anova_results.round(4))
        posthoc = pg.pairwise_tests(dv='expl_time', between='group',within='Task', subject='MouseID', data=anova_data, parametric=True, padjust='Holm')
        print(posthoc)
    else:
        kruskal = pg.kruskal(dv='expl_time', between='group',  data=anova_data)
        print(kruskal)
        posthoc = pg.pairwise_tests(dv='expl_time', between='group', subject='MouseID', data=anova_data, parametric=False)
    
    return(posthoc)
#not used
def run_exploration_stats(df):
    results = {}


    # Extract object columns: columns matching pattern 'Object*_min*'
    obj_cols = [col for col in df.columns if col.startswith('Object') and '_min6' in col]

    if not obj_cols:
        raise ValueError("No object exploration columns found matching 'Object*_min*' pattern.")

    # Convert wide to long format: one row per Subject, group, Task, Object, ExplorationTime
    long_df = df.melt(id_vars=['MouseID', 'group', 'Task'],
                      value_vars=obj_cols,
                      var_name='Object_min',
                      value_name='ExplorationTime')

    # Extract Object and Phase/min from 'Object_min' (e.g., 'Object1_min6')
    long_df[['Object', 'Phase_min']] = long_df['Object_min'].str.split('_', expand=True)
    # Optional: if you want phase/min as numeric, parse from Phase_min (e.g., 'min6' -> 6)
    long_df['Phase'] = long_df['Phase_min'].str.extract('min(\d+)').astype(int)

    # 1) Normality test per Task (Phase), group
    for task in long_df['Task'].unique():
        results[task] = {}
        for grp in long_df['group'].unique():
            sub = long_df[(long_df['Task'] == task) & (long_df['group'] == grp)]
            # Test normality on exploration times (all objects pooled here)
            data = sub['ExplorationTime'].dropna()
            if len(data) >= 3:
                stat, p = shapiro(data)
                results[task][f'shapiro_{grp}'] = (stat, p)
            else:
                results[task][f'shapiro_{grp}'] = ('too few values', None)

    # 2) Levene's test for equal variances per Task between groups
    for task in long_df['Task'].unique():
        task_data = long_df[long_df['Task'] == task]
        groups = task_data['group'].unique()
        if len(groups) == 2:
            g1_data = task_data[task_data['group'] == groups[0]]['ExplorationTime'].dropna()
            g2_data = task_data[task_data['group'] == groups[1]]['ExplorationTime'].dropna()
            if len(g1_data) > 0 and len(g2_data) > 0:
                lev_stat, lev_p = levene(g1_data, g2_data)
                results[task]['levene'] = (lev_stat, lev_p)

    # 3) Total exploration time over all phases (sum all object cols per Subject, group)
    for task in df['Task'].unique():
        task_data = df[df['Task'] == task]
        total_df = task_data.groupby(['MouseID', 'group'], as_index=False)['TotalExploration'].sum()
        
        groups = total_df['group'].unique()
        if len(groups) == 2:
            g1 = total_df[total_df['group'] == groups[0]]['TotalExploration']
            g2 = total_df[total_df['group'] == groups[1]]['TotalExploration']
            
            normal_g1 = shapiro(g1)[1] > 0.05
            normal_g2 = shapiro(g2)[1] > 0.05
            equal_var = levene(g1, g2)[1] > 0.05
    
            if normal_g1 and normal_g2:
                t_stat, p_val = ttest_ind(g1, g2, equal_var=equal_var)
                test_used = 'ttest_ind'
            else:
                t_stat, p_val = mannwhitneyu(g1, g2)
                test_used = 'mannwhitneyu'
    
            results.setdefault('total_exploration', []).append({'task': task, 'test': test_used, 'p': np.round(p_val, 4)})

    # 4) Check if all objects were explored equally per Study Phase and group
    # That is, for each Task, group, compare exploration times across objects

    for task in long_df['Task'].unique():
        for grp in long_df['group'].unique():
            subset = long_df[(long_df['Task'] == task) & (long_df['group'] == grp)]
            if subset.empty:
                continue

            # Pivot to have subjects as rows, objects as columns
            pivot = subset.pivot(index='MouseID', columns='Object', values='ExplorationTime')

            if pivot.isnull().values.any():
                results[f'{task}_{grp}_object_test'] = 'Missing data for some objects'
                continue

            # Check normality per object column
            normal = all(shapiro(pivot[obj])[1] > 0.05 for obj in pivot.columns)

            if normal:
                stat, p = f_oneway(*(pivot[obj] for obj in pivot.columns))
                results[f'{task}_{grp}_object_test'] = ('ANOVA', np.round(p,4))
            else:
                stat, p = kruskal(*(pivot[obj] for obj in pivot.columns))
                results[f'{task}_{grp}_object_test'] = ('Kruskal-Wallis', np.round(p,4))

    return results



#%% Compute disc ratios oer all tests    
def all_tests(path, minute, groups):
    """groups == True for youngs, groups == 'old' for aged mice"""
    df= pd.read_csv(path+'df_min.csv', index_col=0, header=0)
    df=df.sort_values('Task')
    #df['group'] = df['group'].replace('Chrim', 'primed')
    df= df.drop(df.loc[df.MouseID=='B6J-7037'].index)
    #df= df.drop(df.loc[df.MouseID=='B6J-7683'].index)
    if groups== True:
        """Individual polts over Tests"""
        num_chrim= len(df['MouseID'].loc[(df.group=='Chrim')].unique())
        print(df['MouseID'].loc[(df.group=='Control')].unique())
        num_chrim= num_con= len(df['MouseID'].loc[(df.group=='Control')].unique())
        plt.figure()
        sns.lineplot(data=df.loc[df.Task.str.contains('Test')], y= df.loc[df.Task.str.contains('Test'), 'disc_ratio_min'+str(minute)],x='Task', hue= 'group', hue_order=['Control', 'Chrim'], palette= ['dimgrey','darkorange'])
        #sns.lineplot(data=df.loc[(df.Task.str.contains('Test')) & (df.group=='Chrim')], y= df.loc[df.Task.str.contains('Test'), 'disc_ratio_min'+str(minute)],x='Task', hue= 'MouseID', palette=['navajowhite']*num_chrim)
        #sns.lineplot(data=df.loc[(df.Task.str.contains('Test')) & (df.group=='Control')], y= df.loc[df.Task.str.contains('Test'), 'disc_ratio_min'+str(minute)],x='Task', hue= 'MouseID', palette=['gainsboro']*num_con)
        plt.title('cumulative disc ratio in min'+str(minute))
        #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #plt.axhline(0, 0, color= 'k')
        #plt.grid()
        plt.show()
        
        ax=plt.subplot(2,1,1)
        sns.lineplot(data=df.loc[(df.Task.str.contains('Test')) & (df.group=='Control')], y= df['disc_ratio_min'+str(minute)],x='Task', hue= 'MouseID')
        plt.ylabel('disc ratio min'+str(minute))
        plt.title('Control mice')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.axhline(0, 0, color= 'k')
        plt.grid()
        plt.subplot(2,1,2, sharey=ax)
        sns.lineplot(data=df.loc[df.Task.str.contains('Test') & (df.group=='Chrim')], y= df['disc_ratio_min'+str(minute)],x='Task', hue= 'MouseID')
        
        plt.ylabel('disc ratio min'+str(minute))
        plt.title('Primed mice')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.axhline(0, 0, color= 'k')
        plt.grid()
        plt.tight_layout()
       #plt.savefig(path+'allTest_ind_min'+str(minute)+'.png')
        
        """Boxplots over tests"""

        # Figure setup
        plt.figure(figsize=(4,3.54), dpi=600)
        ax = plt.subplot()
        
        # Boxplot
        sns.boxplot(data=df.loc[df.Task.str.contains('Test')],
                    y=df.loc[df.Task.str.contains('Test'), 'disc_ratio_min'+str(minute)],
                    x='Task', hue='group', hue_order=['Control', 'Chrim'],
                    palette={'Control':'dimgray','Chrim':'darkorange'},
                    saturation=1, boxprops=dict(alpha=.45), width=0.4)
        
        # Stripplot
        sns.stripplot(data=df.loc[df.Task.str.contains('Test')],
                      y=df.loc[df.Task.str.contains('Test'), 'disc_ratio_min'+str(minute)],
                      x='Task', hue='group', hue_order=['Control', 'Chrim'],
                      palette=['dimgrey','darkorange'], dodge=True, ax=ax,
                      size=5.5, alpha=.8, edgecolor='k', linewidth=1, legend=False)
        handles, labels = ax.get_legend_handles_labels()
        labels = ['Control' if lbl == 'Control' else 'Primed' for lbl in labels]
        ax.legend(handles=handles[:2], labels=labels[:2], title='Group')  # [:2] to avoid duplicate legend entries
        
        print("\n=== Mixed ANOVA ===")#not used because of missing data
        aov = pg.mixed_anova(dv='disc_ratio_min'+str(minute), within='Task', between='group', subject='MouseID', data=df.loc[df.Task.str.contains('Test')])
        print(aov)
        # Post-hoc: Pairwise Wilcoxon tests for within-subject comparisons
        post = pg.pairwise_tests(dv='disc_ratio_min' + str(minute), within='Task', between='group', subject='MouseID', 
                                 data=df.loc[df.Task.str.contains('Test')], parametric=False, padjust='fdr_bh')
        
        print(post)
        
        f= pg.rm_anova(dv='disc_ratio_min'+str(minute), within='Task', subject='MouseID', data=df.loc[(df.Task.str.contains('Test'))&(df.group.str.contains('Chrim'))])
        ff= pg.friedman(dv='disc_ratio_min'+str(minute), within='Task', subject='MouseID', data=df.loc[(df.Task.str.contains('Test'))&(df.group.str.contains('Chrim'))])
        
        print('Anova Chrim', f)
        print('Friedman Chrim', ff)
        f= pg.friedman(dv='disc_ratio_min'+str(minute), within='Task', 
                         subject='MouseID', data=df.loc[(df.Task.str.contains('Test'))&(df.group.str.contains('Control'))])
        print('Friedman Control', f)
        
        # **Test against 0 for both groups**
        unique_tasks = df.loc[df.Task.str.contains('Test'), 'Task'].unique().tolist()

        for task in unique_tasks:
           print(f"\nPerforming Wilcoxon Signed-Rank/ttest Test for {task}")

           # Filter data for the specific task
           task_data = df.loc[df.Task == task]
        
           # Control group for this task
           control_data = task_data.loc[task_data.group == 'Control', 'disc_ratio_min' + str(minute)]
           
           stat_shap, p_shap = shapiro(control_data)
           print('shapiro for control',task, stat_shap, p_shap)
           # Wilcoxon test: Test against 0 for the Control group
           if len(control_data) > 1:  # Ensure enough data
               stat_control, p_value_control = wilcoxon(control_data - 0)  # Test against 0
               print(f"Control group Wilcoxon test for {task} against 0 : p = {p_value_control}")
               t_stat, p_value = stats.ttest_1samp(control_data, 0)
               print(f"Ttest Control group ttest for {task} against 0 : p = {p_value}")
           else:
               print(f"Not enough data in the Control group for {task}")
        
           # Chrim group for this task
           chrim_data = task_data.loc[task_data.group == 'Chrim', 'disc_ratio_min' + str(minute)]
           
           stat_shap, p_shap = shapiro(chrim_data)
           print('shapiro for chrim',task, stat_shap, p_shap)
        # Wilcoxon test: Test against 0 for the Chrim group
           if len(chrim_data) > 1:  # Ensure enough data
               stat_chrim, p_value_chrim = wilcoxon(chrim_data - 0)  # Test against 0
               print(f"Chrim group Wilcoxon test for {task} against 0 : p = {p_value_chrim}")
               t_stat, p_value = stats.ttest_1samp(chrim_data, 0)
               print(f"Ttest Chrim group test for {task} against 0 : p = {p_value}")
           else:
               print(f"Not enough data in the Chrim group for {task}")
        

        # Extract p-values from post-hoc tests
        # Extract only **within-task** group comparisons (Control vs Chrim for each task)
        sig_tests = post[(post['Contrast'] == 'Task * group') & 
                 (((post['A'] == 'Control') & (post['B'] == 'Chrim')) |
                  ((post['A'] == 'Chrim') & (post['B'] == 'Control')))].copy()
        sig_tests['Task'] = post.loc[sig_tests.index, 'Task']  # Extract task names correctly
        print(sig_tests)
        # Extract the order of tasks from the boxplot
        task_order = df.loc[df.Task.str.contains('Test'), 'Task'].unique().tolist()
        
        # Add significance stars
        y_max = df['disc_ratio_min' + str(minute)].max()
        y_offset = y_max * 0.05  # Space above the box for stars
    
        for i, row in sig_tests.iterrows():
            task = row['Task']
            p_val = row['p-unc']
            
            if p_val < 0.001:
                star = "***"
            elif p_val < 0.01:
                star = "**"
            elif p_val < 0.05:
                star = "*"
            else:
                continue  # Skip non-significant comparisons
    
            # Ensure both tasks exist in the order list
            if task not in task_order:
                print(f"Warning: Tasks '{task}' not found in task_order. Skipping.")
                continue
    
            # Get x positions
            x_pos= task_order.index(task)
            
            max_y = df.loc[df.Task == task, 'disc_ratio_min' + str(minute)].max()
            print(x_pos)
            plt.text(x_pos, max_y + y_offset, star, ha='center', fontsize=12, fontweight='bold', color='black')
    
        
        
        # Formatting
        plt.ylabel('Disc ratio min ' + str(minute))
        plt.title('Cumulative disc ratio in min ' + str(minute))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.tight_layout()
        
        # Save figure (optional)
        plt.savefig(path+'allTest_min' + str(minute) + '.svg')
        
        plt.show()

    elif groups== 'old':
        """Boxplots over tests"""
        #df['group_age']=df['group']+'_'+df['age']
        plt.figure(figsize=(4,3.54), dpi=600)
        ax= plt.subplot()
        sns.boxplot(data=df.loc[(df.Task.str.contains('Test'))], y= df.loc[df.Task.str.contains('Test'), 'disc_ratio_min'+str(minute)],x='Task', hue= df[['group','age']].apply(tuple,axis=1), hue_order=[('Control', 'young'), ('Control', 'old'), ('Chrim', 'young'), ('Chrim', 'old')], palette= {('Control', 'young'):'dimgray',('Control', 'old'):'black', ('Chrim', 'young'):'sandybrown', ('Chrim', 'old'):'darkorange'}, saturation=1, boxprops=dict(alpha=.6), width=0.4)
        #sns.stripplot(data=df.loc[(df.Task.str.contains('Test'))], y= df.loc[df.Task.str.contains('Test'), 'disc_ratio_min'+str(minute)],x='Task', hue= df[['group','age']].apply(tuple,axis=1), hue_order=[('Control', 'young'), ('Control', 'old'), ('Chrim', 'young'), ('Chrim', 'old')], palette= {('Control', 'young'):'dimgray',('Control', 'old'):'black', ('Chrim', 'young'):'sandybrown', ('Chrim', 'old'):'darkorange'}, 
         #              dodge=True, ax=ax,size=5.5, alpha=.6, edgecolor='k', linewidth=1, legend= False)
        
        
        plt.ylabel('disc ratio min'+str(minute))
        plt.title('cumulative disc ratio in min'+str(minute))
        #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #plt.axhline(0, 0, color= 'k')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.tight_layout()
        plt.savefig(path+'allTest_min'+str(minute)+'.svg')
        
        #Repeated Anova
        aov= pg.mixed_anova(dv= 'disc_ratio_min'+str(minute), within='Task', between='group', subject= 'MouseID', data=df.loc[df.Task.str.contains('Test')])
        print(aov)
        post= pg.pairwise_ttests(dv= 'disc_ratio_min'+str(minute), within='Task', between= 'group', subject='MouseID', data= df.loc[(df.Task.str.contains('Test'))])
    
    
            
from scipy.stats import shapiro, levene



def stat_oldChrim(path, minute):
    # Load and clean data
    df = pd.read_csv(path + 'df_min.csv', index_col=0, header=0)
    df = df.drop(df.loc[df.MouseID == 'B6J-7037'].index)
    df = df[df.Task.str.contains('Test')]
    dv = 'disc_ratio_min' + str(minute)

    print(f"\n=== Mann–Whitney U tests between groups per Task for {dv} ===\n")
    for task in sorted(df['Task'].unique()):
        sub = df[df['Task'] == task]
        group1 = sub[sub['group'] == 'Control'][dv]
        group2 = sub[sub['group'] == 'Chrim'][dv]
        
        if len(group1) >= 2 and len(group2) >= 2:
            stat, p = mannwhitneyu(group1, group2, alternative='two-sided')
            print(f"Task: {task}, U = {stat:.2f}, p = {p:.4f}, n1 = {len(group1)}, n2 = {len(group2)}")
        else:
            print(f"Task: {task} - Not enough data for both groups")

    print(f"\n=== Wilcoxon signed-rank test within group over tasks for {dv} ===\n")
    for group in ['Control', 'Chrim']:
        print(f"\nGroup: {group}")
        # Pivot to get repeated measures per mouse
        df_group = df[df['group'] == group].pivot(index='MouseID', columns='Task', values=dv)
        
        tasks = sorted(df_group.columns)
        if len(tasks) < 2:
            print("Not enough tasks for comparison.")
            continue
        
        for i in range(len(tasks) - 1):
            t1, t2 = tasks[i], tasks[i + 1]
            paired_data = df_group[[t1, t2]].dropna()
            if len(paired_data) >= 2:
                stat, p = wilcoxon(paired_data[t1], paired_data[t2], alternative='two-sided')
                print(f"{t1} vs {t2}: W = {stat:.2f}, p = {p:.4f}, n = {len(paired_data)}")
            else:
                print(f"{t1} vs {t2}: Not enough paired data")
        
        # Explicitly test first vs last task
        if len(tasks) >= 3:
            t_first, t_last = tasks[0], tasks[2]  # assuming OIP1 = first, OIP3 = third
            paired_data = df_group[[t_first, t_last]].dropna()
            if len(paired_data) >= 2:
                stat, p = wilcoxon(paired_data[t_first], paired_data[t_last], alternative='two-sided')
                print(f"{t_first} vs {t_last}: W = {stat:.2f}, p = {p:.4f}, n = {len(paired_data)}")
            else:
                print(f"{t_first} vs {t_last}: Not enough paired data")


def run_tests(path, minute):
    """statistical tests separately"""
    df = pd.read_csv(path + 'df_min.csv', index_col=0)
    df = df.drop(df.loc[df.MouseID == 'B6J-7037'].index)
    df = df[df.Task.str.contains('Test')]
    df['Task'] = df['Task'].astype('category')

    dv = 'disc_ratio_min' + str(minute)
    group_col = 'group'
    subject_col = 'MouseID'
    # Pivot table for normality/friedman
    df_wide = df.pivot(index=subject_col, columns='Task', values=dv).dropna()
    
    print("\n=== epeated Measures ANOVA and Friedman per Group ===")
    for group in df[group_col].unique():
        print(f"\nGroup: {group}")
        df_group = df[df[group_col] == group]
        # Check for enough data per subject
        df_wide_group = df_group.pivot(index=subject_col, columns='Task', values=dv).dropna()
        if df_wide_group.shape[0] < 3:
            print("  Not enough complete data for repeated-measures ANOVA.")
            continue
        try:
            aov_group = pg.rm_anova(dv=dv, within='Task', subject=subject_col, data=df_group, detailed=True)
            stat, p = friedmanchisquare(*[df_wide_group[col] for col in df_wide_group.columns])
            print(f"χ² = {stat:.3f}, p = {p:.4f}")
            print(aov_group)
        except Exception as e:
            print("  Error in rm_anova:", e)
    
    
    
    print("=== Shapiro-Wilk Test for Normality (per group and Task) ===")
    normality = {}
    for group in df[group_col].unique():
        for task in df['Task'].unique():
            subset = df[(df[group_col] == group) & (df['Task'] == task)][dv].dropna()
            if len(subset) >= 3:  # Shapiro requires at least 3 values
                p = shapiro(subset)[1]
                normality[(group, task)] = p
                print(f"{group} - {task}: p = {p:.4f}")
            else:
                print(f"{group} - {task}: Not enough data for Shapiro-Wilk")
                normality[(group, task)] = None

    normal_data = all(p is None or p > 0.05 for p in normality.values())

    print("\n=== Levene's Test for Equal Variance (between groups) ===")
    for task in df['Task'].unique():
        task_df = df[df.Task == task]
        groups = [group[dv].dropna().values for name, group in task_df.groupby(group_col)]
        if len(groups) > 1:
            _, p_levene = levene(*groups)
            print(f"Levene test for {task}: p = {p_levene:.3f}")
    
    print("\n=== Mixed ANOVA ===")#not used because of missing data
    aov = pg.mixed_anova(dv=dv, within='Task', between=group_col, subject=subject_col, data=df)
    print(aov)
    post = pg.pairwise_tests(dv='disc_ratio_min' + str(minute), within='Task', between='group', subject='MouseID', 
                             data=df.loc[df.Task.str.contains('Test')], parametric=True, padjust='fdr_bh')
    print(post)
    if normal_data:
        print("\n=== Mixed ANOVA ===")#not used because of missing data
        aov = pg.mixed_anova(dv=dv, within='Task', between=group_col, subject=subject_col, data=df)
        print(aov)
        post = pg.pairwise_tests(dv='disc_ratio_min' + str(minute), within='Task', between='group', subject='MouseID', 
                                 data=df.loc[df.Task.str.contains('Test')], parametric=True, padjust='fdr_bh')
        
        print(post)
        
            # Kendall's W effect size
            
            #k_w = pg.compute_effsize(df_wide_group.group)
            #print(f"{group}: Kendall's W = {k_w:.3f}")
    else:
        print("\n=== Kruskal Test ===")
        # Perform separately for each group
        for group in df[group_col].unique():
            # Step 1: Compute mean or median per group per task
            group_summary = df.groupby([group_col, 'Task'])[dv].median().reset_index()  # or .mean()
            
            # Step 2: Pivot: rows = groups, columns = tasks
            group_wide = group_summary.pivot(index=group_col, columns='Task', values=dv).dropna()
            print(group_wide)
            # Step 3: Friedman test across tasks using group summaries
            stat, p = kruskal(group_summary[g] for g in group_summary.Task)#?
            
            print("\n=== Kruskal-Wallis Test Across Tasks (per Group) ===")
            print(f"χ² = {stat:.3f}, p = {p:.4f}")
            
            # Step 4: Kendall's W (effect size)
            #kendalls_w = pg.compute_effsize(group_wide, eftype='kendall')
            #print(f"Kendall's W = {kendalls_w:.3f}")
            
            # Optional: Display group-wise task values
            print("\nGroup-level task medians:")
            print(group_wide)

    print("\n=== Pairwise Post-Hoc Comparisons ===")
    posthoc = pg.pairwise_tests(dv=dv, within='Task', between=group_col, subject=subject_col,
                                 data=df, parametric=normal_data, padjust='fdr_bh', effsize='r')#'hedges' if normal_data else 'r')
    print(posthoc)

    return posthoc
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

def LMM(path, minute):
    """linear mixed model"""
    df = pd.read_csv(path + 'df_min.csv', index_col=0)
    df = df.drop(df.loc[df.MouseID == 'B6J-7037'].index)
    df = df[df.Task.str.contains('Test')]
    df['Task'] = df['Task'].astype('category')
    
    # Define dependent variable
    dv = 'disc_ratio_min' + str(minute)
    
    # Fit linear mixed model
    model = smf.mixedlm(f"{dv} ~ Task * group", data=df, groups=df["MouseID"])
    result = model.fit()
    
    print(result.summary())
    model = smf.ols('disc_ratio_min3 ~ Task * group', data=df).fit()

    # ANOVA table
    anova_results = anova_lm(model, typ=2)
    print('Result Anova:', anova_results)
    
    df['condition'] = df['group'].astype(str) + '_' + df['Task'].astype(str)

    # Make sure it's in the DataFrame
    #print(df.columns)  # Debugging step
    """
    # Now run the post hoc test (not needed)
    posthoc = pg.pairwise_tests(
        data=df,
        dv=dv,
        between='condition',
        padjust='holm',
        effsize='hedges'
    )
        
    print(posthoc[['A', 'B', 'p-corr']])"""

    return result, anova_results#, posthoc
    
#%% DLC analysis, dist traveled, heatmaps chi2
def process_df(filename):
  data= pd.read_csv(filename, header=[1,2])
  
  #to avoid settings with copy warning
  data_Nose = data['Nose'].copy()
  data_LeftFront = data['FiberLeft'].copy()
  data_RightFront = data['FiberRight'].copy()
  data_TailBase = data["Tailbase"].copy()
  
  
  #drop less likely data
  data_Nose.drop(data_Nose[data_Nose['likelihood']<=0.9].index, inplace= True)
  
  data_LeftFront.drop(data_LeftFront[data_LeftFront['likelihood']<=0.9].index, inplace= True)

  data_RightFront.drop(data_RightFront[data_RightFront['likelihood']<=0.9].index, inplace= True)
 
  data_TailBase.loc[data_TailBase['likelihood'] <= 0.9, ['x', 'y']] = np.nan

    # **Apply interpolation to fill gaps**
  for part in [data_Nose, data_LeftFront, data_RightFront, data_TailBase]:
        part[['x', 'y']] = part[['x', 'y']].interpolate(method='linear', limit_direction='both')

  #put it back to dataframe
  data['Nose']=data_Nose
  data['FiberRight']= data_RightFront
  #data['FiberRight']= data_RightFront
  data['FiberLeft']= data_LeftFront
  #data['ShoulderLeft']= data_LeftHind
  data['Tailbase'] = data_TailBase

  return(data)

def process_df(filename):
  data= pd.read_csv(filename, header=[1,2])
  
  #to avoid settings with copy warning
  data_Nose = data['Nose'].copy()
  data_LeftFront = data['FiberLeft'].copy()
  data_RightFront = data['FiberRight'].copy()
  data_TailBase = data["TailBase"].copy()
  
  
  #drop less likely data
  data_Nose.drop(data_Nose[data_Nose['likelihood']<=0.9].index, inplace= True)
  
  data_LeftFront.drop(data_LeftFront[data_LeftFront['likelihood']<=0.9].index, inplace= True)

  data_RightFront.drop(data_RightFront[data_RightFront['likelihood']<=0.9].index, inplace= True)
 
  data_TailBase.loc[data_TailBase['likelihood'] <= 0.9, ['x', 'y']] = np.nan

    # **Apply interpolation to fill gaps**
  for part in [data_Nose, data_LeftFront, data_RightFront, data_TailBase]:
        part[['x', 'y']] = part[['x', 'y']].interpolate(method='linear', limit_direction='both')

  #put it back to dataframe
  data['Nose']=data_Nose
  data['FiberRight']= data_RightFront
  #data['FiberRight']= data_RightFront
  data['FiberLeft']= data_LeftFront
  #data['ShoulderLeft']= data_LeftHind
  
  data['TailBase'] = data_TailBase

  return(data)
ending= 'DLC_resnet50_JULIA_DLCJul25shuffle1_940000.csv'
endingFiber= 'OIP1/DLCFibernet/','DLC_resnet50_OIPFiberNetDec30shuffle1_105000.csv'

def track_length(path_to_test, ending):
    """For Fibernet Tailbase for JuliaDLC TailBase also need to be changed for process_df"""
    animal = []
    proced = []
    px_cm = 365/31.5  #365/31.5
   # moving_avg_window=5
    
    # Get animal and procedure names
    for filename in os.listdir(path_to_test): 
        if filename.endswith(ending):
            animal.append(filename.split('_')[0]+'_')
            p = filename.split('_')[1]
            proced.append(p.split('D')[0])

    trav_dist = pd.DataFrame(index=np.arange(len(animal)), columns=['MouseID', 'Task', 'dist_min1', 'dist_min2', 'dist_min3', 'dist_min4', 'dist_min5', 'dist_min6'])
    
    for idx, ani in enumerate(animal):
        print('Analyzing ', ani, proced[idx])
        dist_trav = trav_dist.iloc[idx]
        dist_trav['MouseID'] = ani
        dist_trav['Task'] = proced[idx]
        df = process_df(path_to_test + ani + proced[idx] + ending)
        
        
        dist = [0] * 6
        
        # Iterate through frames in the video
        for idx, x in enumerate(df['TailBase', 'x'].dropna()):
            if idx > 0:
                if np.isnan(df['TailBase', 'x'][idx]) or np.isnan(df['TailBase', 'x'][idx - 1]):
                    continue

                y = df['TailBase', 'y'].dropna()[idx]
                x1 = df['TailBase', 'x'][idx - 1]
                y1 = df['TailBase', 'y'][idx - 1]
                
                # Calculate Euclidean distance in cm
                dist_xy = abs((x1 - x)**2 + (y1 - y)**2)**0.5 / px_cm
                velocity = dist_xy / (1 / 30)  # velocity in cm/s (assuming 30 fps)

                # Debugging: Check distance and velocity
               # print(f"Frame {idx}: dist_xy = {dist_xy:.2f} cm, velocity = {velocity:.2f} cm/s")

                # Skip frames with slow velocity (likely grooming or no movement)
                if velocity < 2 or velocity > 200:  # Filter for velocity below 2 cm/s
                    continue
                
                # Skip frames with minimal and maxiamal displacement (likely noise or grooming)
                if dist_xy < 0.1:  # Adjust threshold for minimal movement
                    continue
                elif dist_xy > 20:
                    continue
                
            
                # Add the distance to the appropriate time interval based on the frame index
                if idx < 1 * 30 * 60:  # Minute 1
                    dist[0] += dist_xy
                elif 1 * 30 * 60 < idx < 2 * 30 * 60:  # Minute 2
                    dist[1] += dist_xy
                elif 2 * 30 * 60 < idx < 3 * 30 * 60:  # Minute 3
                    dist[2] += dist_xy
                elif 3 * 30 * 60 < idx < 4 * 30 * 60:  # Minute 4
                    dist[3] += dist_xy
                elif 4 * 30 * 60 < idx < 5 * 30 * 60:  # Minute 5
                    dist[4] += dist_xy
                elif 5 * 30 * 60 < idx < 6 * 30 * 60:  # Minute 6
                    dist[5] += dist_xy

        #dist_smoothed = pd.Series(dist).rolling(window=moving_avg_window, min_periods=1).mean().tolist()
        
        # Assign calculated distances to the DataFrame
        dist_trav[['dist_min1', 'dist_min2', 'dist_min3', 'dist_min4', 'dist_min5', 'dist_min6']] = dist

    # Sum the distances for each task and animal
    trav_dist['sum'] = trav_dist.loc[:, 'dist_min1':'dist_min6'].sum(axis=1)

    # Visualize the data
    sns.stripplot(data=trav_dist, x='MouseID', y='sum', hue='Task')
    plt.xticks(rotation=45)
    plt.figure()
    plt.plot(range(1, 7), trav_dist.loc[:, trav_dist.columns.str.contains('min')].T, label=trav_dist.MouseID)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid()
    plt.xlabel('minute')
    plt.ylabel('distance traveled in [cm]')
    plt.savefig(path_to_test + 'dist_permin', bbox_inches='tight')

    # Update or create the track.csv file
    """path_cohort = os.path.dirname(os.path.dirname(os.path.dirname(path_to_test)))
    if os.path.exists(path_cohort + '/track1.csv'):
        print('track1.csv is updated')
        track = pd.read_csv(path_cohort + '/track1.csv', index_col=[0])
        track = pd.concat([track, trav_dist.loc[:, ['MouseID', 'Task', 'sum']]], ignore_index=True)
        track.to_csv(path_cohort + '/track1.csv', index=False)
    else:
        print('track1.csv is created in ' + path_cohort + '/track1.csv')
        track = trav_dist.loc[:, ['MouseID', 'Task', 'sum']]
        track.to_csv(path_cohort + '/track1.csv', index=False)"""
    path_cohort = os.path.dirname(os.path.dirname(os.path.dirname(path_to_test)))
    new_data = trav_dist.loc[:, ['MouseID', 'Task', 'sum']]
    
    # Check if the file exists
    if os.path.exists(path_cohort + '/track1.csv'):
        print('Appending to track1.csv')
        
        # Append without modifying existing data
        new_data.to_csv(path_cohort + '/track1.csv', mode='a', header=False, index=False)
    
    else:
        print(f'Creating track1.csv in {path_cohort}')
        
        # Write with header only if the file is created
        new_data.to_csv(path_cohort + '/track1.csv', index=False)

    return new_data

import statsmodels.formula.api as smf
def test_normality(data, dv='sum', group='group', subject='MouseID', within='Task', phase=''):
    print(f"\nNormality check for residuals – {phase} phase")
    data = data.copy()
    try:
        # Build linear model (simplified, ignoring interaction for residuals)
        model = smf.ols(f'{dv} ~ C({group}) * C({within})', data=data).fit()
        residuals = model.resid
        stat, p = shapiro(residuals)
        print(f'Shapiro-Wilk Test for residuals: W={stat:.3f}, p={p:.4f}')
        if p > 0.05:
            print("→ Residuals are likely normal.")
        else:
            print("→ Residuals deviate from normality.")
    except Exception as e:
        print(f"Could not compute normality: {e}")
#%%plots for tracklength, heatmaps etc 
from statsmodels.stats.multitest import multipletests
def plot_dist_trav(path_cohort, old= False):

    # Load data
    track = pd.read_csv(path_cohort + 'track.csv')
    df_min = pd.read_csv(path_cohort + 'df_min.csv', index_col=[0])
    print(track.head())
    # Clean and preprocess
    df_min = df_min.drop(df_min.loc[df_min.MouseID.isin(['B6J-7037_', 'B6J-6678'])].index)
    track = track.drop(track.loc[track.MouseID.isin(['B6J-7037_', 'B6J-6678_'])].index)
    #track['group'] = pd.Series(dtype='string')
    #df_min['track'] = pd.Series()
    track['MouseID'] = track['MouseID'].str.rstrip('_')
    
    # Ensure consistent MouseID formatting
    df_min['MouseID'] = df_min['MouseID'].astype(str).str.strip().str.lower()
    track['MouseID'] = track['MouseID'].astype(str).str.strip().str.lower()

    # Reset indices for both dataframes
    df_min = df_min.reset_index(drop=True)
    track = track.reset_index(drop=True)

    # Merge the 'group' information from df_min into track
    #track = track.merge(df_min[['MouseID', 'group']], on='MouseID', how='left')
    track = track.merge(df_min[['MouseID', 'group']].drop_duplicates(), on='MouseID', how='left')
    print(track.head())
    # Merge track data into df_min based on MouseID and Task
    df_min = df_min.merge(track[['MouseID', 'sum']].drop_duplicates(), on='MouseID', how='left')

    # Handle missing data (where track sum might be missing)
    df_min['track'] = df_min['sum'].fillna(np.nan)

    # Drop the temporary 'sum' column
    df_min = df_min.drop(columns=['sum'])

    # Boxplots over tests
    boxprops = dict(alpha=.45)  # Define boxprops variable before usage

    plt.figure(figsize=(4, 3.54), dpi=300)
    ax = plt.subplot()
   
    # Ensure the data passed to sns.boxplot is valid
    test_data = track[~track.Task.str.contains('Hab')]
    if test_data.empty:
        print("No valid test data found.")
    else:
        sns.boxplot(data=test_data, y='sum', x='Task', hue='group', hue_order=['Control', 'Chrim'],
                    palette={'Control': 'dimgray', 'Chrim': 'darkorange'},
                    order=sorted(test_data.Task.unique()), saturation=1, boxprops= dict(alpha=.45), width=0.4)
        #sns.stripplot(data=test_data, y='sum', x='Task', hue='group', order=sorted(test_data.Task.unique()),
         #             palette={'Control': 'dimgray', 'Chrim': 'darkorange', 'ChR2': 'lightblue'}, dodge=True, ax=ax,
          #            size=5.5, alpha=.8, edgecolor='k', linewidth=1, legend=False)

    plt.ylabel('distance traveled [cm]')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig(path_cohort + 'track_test.svg')
    
    if old== True:
        plt.figure()
        ax = plt.subplot()
        test_data= test_data.merge(df_min[['MouseID', 'age']], on='MouseID', how='left')
        print(test_data.head())
        sns.boxplot(data=test_data, y='sum', x='Task', hue=test_data[['group','age']].apply(tuple,axis=1), hue_order=[('Control', 'young'), ('Control', 'old'), ('Chrim', 'young'), ('Chrim', 'old')], 
                    palette= {('Control', 'young'):'dimgray',('Control', 'old'):'black', ('Chrim', 'young'):'sandybrown', ('Chrim', 'old'):'darkorange'}, saturation=1, boxprops=dict(alpha=.6), width=0.4,
                    order=sorted(test_data.Task.unique()))#, saturation=1, boxprops= dict(alpha=.45), width=0.4)
        #sns.stripplot(data=test_data, y='sum', x='Task', hue='group', order=sorted(test_data.Task.unique()),
         #             palette={'Control': 'dimgray', 'Chrim': 'darkorange', 'ChR2': 'lightblue'}, dodge=True, ax=ax,
          #            size=5.5, alpha=.8, edgecolor='k', linewidth=1, legend=False)

        plt.ylabel('distance traveled [cm]')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.tight_layout()
        plt.savefig(path_cohort + 'track_test.svg')
    # Habituation plot
    

    # Ensure the data passed to sns.boxplot is valid
    hab_data = track[track.Task.str.contains('Hab')]
    if hab_data.empty:
        print("No valid habituation data found.")
    else:
        hab_data['Task'] = hab_data['Task'].replace({
    'ObjectHab1.1': 'HabObj1.1',
    'ObjectHab1.2': 'HabObj1.2',
    'ObjectHab2.1': 'HabObj2.1',
    'ObjectHab2.2': 'HabObj2.2'
    })
    plt.figure(figsize=(5, 3.54), dpi=300)
    ax = plt.subplot()
    sns.boxplot(data=hab_data, y='sum', x='Task', hue='group', hue_order=['Control', 'Chrim'],palette={'Control': 'dimgray', 'Chrim': 'darkorange'}, 
                order=['ArenaHab1.1', 'ArenaHab1.2', 'ArenaHab2.1', 'ArenaHab2.2', 'HabObj1.1', 'HabObj1.2', 'HabObj2.1', 'HabObj2.2', 'Hab2.1', 'Hab2.2', 'Hab3.1', 'Hab3.2'],
                saturation=1, boxprops=boxprops, width=0.4)
    #sns.stripplot(data=hab_data, y='sum', x='Task', hue='group', hue_order=['Control', 'ChR2','Chrim'],
     #             order=sorted(hab_data.Task.unique()), palette={'Control': 'dimgray', 'Chrim': 'darkorange', 'ChR2': 'lightblue'}, dodge=True, ax=ax,
      #            size=5.5, alpha=.8, edgecolor='k', linewidth=1, legend=False)

    plt.ylabel('distance traveled [cm]')
    plt.xticks(rotation=45, ha='right')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    labels = ['Control' if lbl == 'Control' else 'Primed' for lbl in labels]
    ax.legend(handles=handles[:2], labels=labels[:2], title='Group')
    plt.tight_layout()
    plt.savefig(path_cohort + 'track_Hab.svg')
    
 
    # Disc ratio track length
    plt.figure()
    sns.stripplot(data=df_min[df_min.Task.str.contains('Test')].dropna(), x='track', y='disc_ratio_min3', hue='group')

    # Filter out NaN values before performing ANOVA
    track_clean = track.dropna(subset=['sum', 'Task', 'MouseID', 'group'])

    # Clean test phase data
    #Anova or friedmann tests
    #for phase in ['Test', 'Study', 'Hab']: 
       # test_data = track_clean[track_clean['Task'].str.contains(phase)].dropna(subset=['sum'])
    test_data = track_clean.dropna(subset=['sum'])
    print(test_data)
    # Assumption checks
    normality_results = {}
    #test_data=test_data[test_data['Task'].str.contains('Test|Study')]
    # Test normality for each group separately
    for group in test_data['group'].unique():
        group_data = test_data[test_data['group'] == group]['sum']
        
        if len(group_data) >= 3:  # Shapiro requires at least 3 samples
            stat, p = shapiro(group_data)
            normality_results[group] = p
            print(f"Shapiro-Wilk test for group '{group}': p = {p:.4f}")
        else:
            normality_results[group] = None
            print(f"Group '{group}' has less than 3 samples. Skipping Shapiro-Wilk test.")
    
    # Check if all groups are normally distributed
    all_normal = all(p is not None and p > 0.05 for p in normality_results.values())
    print("\n✅ All groups normal" if all_normal else "❌ At least one group is not normally distributed")
    

    group_values = [g['sum'].values for _, g in test_data.groupby('group')]
    _, p_levene = levene(*group_values)

    # Choose test
    if all_normal and p_levene > 0.05:
        print("Running ANOVA.")
        
        aov = pg.rm_anova(dv='sum', within= 'Task', subject='MouseID', data=test_data.dropna())#pg.mixed_anova(dv='sum', between='group', within='Task', data=test_data.dropna())
        print(aov)
        posthoc = pg.pairwise_tests(dv='sum', between='group', within='Task', subject='MouseID', 
                                    data=test_data, padjust='holm')
    else:
        print("Running Kruskal-Wallis test per group + posthoc.")
        k = pg.kruskal(dv='sum',between='group', data=test_data)
        print(k)
        posthoc = pg.pairwise_tests(dv='sum', between='group', within= 'Task', subject='MouseID', data=test_data, parametric=False)
            # Optional: post hoc with Wilcoxon signed-rank
    

    print('posthoc',posthoc)

    """results = []
    
    for phase in ['Study', 'Test', 'Hab']:
        # Filter for all tasks in this phase (e.g., Study1, Study2)
        phase_data = track_clean[track_clean['Task'].str.contains(phase)].dropna(subset=['sum'])
    
        for task in sorted(phase_data['Task'].unique()):
            task_data = phase_data[phase_data['Task'] == task]
            groups = task_data['group'].unique()
            if len(groups) != 2:
                continue  # skip if not exactly 2 groups
    
            group1_data = task_data[task_data['group'] == groups[0]]['sum']
            group2_data = task_data[task_data['group'] == groups[1]]['sum']
    
            # Normality check
            norm1 = shapiro(group1_data)[1] > 0.05 if len(group1_data) >= 3 else False
            norm2 = shapiro(group2_data)[1] > 0.05 if len(group2_data) >= 3 else False
    
            if norm1 and norm2 and levene(group1_data, group2_data)[1] > 0.05:
                stat, p = ttest_ind(group1_data, group2_data)
                test_type = 't-test'
            else:
                stat, p = mannwhitneyu(group1_data, group2_data, alternative='two-sided')
                test_type = 'Mann–Whitney U'
    
            results.append({
                'phase': phase,
                'task': task,
                'test': test_type,
                'p_uncorrected': p
            })
    
    # Apply multiple comparison correction
    p_vals = [r['p_uncorrected'] for r in results]
    reject, p_corrected, _, _ = multipletests(p_vals, method='bonferroni')  # use 'holm' or 'bonferroni' for stricter control
    
    # Attach corrected p-values
    for i, r in enumerate(results):
        r['p_corrected'] = p_corrected[i]
        r['reject_null'] = reject[i]
    
    # View results
    
    result_df = pd.DataFrame(results)
    print(result_df)"""

    return (hab_data, df_min, posthoc)
    
#%%stats for heatmap (place preference)
# Function to determine the quadrant based on the x, y position
def get_quadrant(x, y, x_mid, y_mid):
    if x <= x_mid and y >= y_mid:
        return 'UL'  # Upper Left
    elif x >= x_mid and y >= y_mid:
        return 'UR'  # Upper Right
    elif x <= x_mid and y <= y_mid:
        return 'LL'  # Lower Left
    elif x >= x_mid and y <= y_mid:
        return 'LR'  # Lower Right
    return None

def calculate_quadrant_time(df, x_mid, y_mid):
    """Calculate the time spent in each quadrant (in seconds)."""
    quadrant_time = {'UL': 0, 'UR': 0, 'LL': 0, 'LR': 0}
    
    # Count the number of frames in each quadrant
    for _, row in df.iterrows():
        x, y = row['FiberLeft']['x'], row['FiberLeft']['y']
        quadrant = get_quadrant(x, y, x_mid, y_mid)
        if quadrant:
            quadrant_time[quadrant] += 1  # Count the frames in the quadrant
    
    # Convert frame counts to seconds (since 30 FPS)
    for quadrant in quadrant_time:
        quadrant_time[quadrant] /= 30  # Convert frames to seconds
    
    return quadrant_time

def chi_square_test(quadrant_time, total_time= None):
    """Perform Chi-square test to compare observed and expected quadrant times."""
    if total_time is None:
       total_time = sum(quadrant_time.values())
   
   # Calculate expected time per quadrant (proportional to the total observed time)
    expected_time_per_quadrant = total_time / 4 
    expected_times = [expected_time_per_quadrant] * 4  # 4 quadrants with equal expected time
    
    observed = [quadrant_time['UL'], quadrant_time['UR'], quadrant_time['LL'], quadrant_time['LR']]
    
    # Perform the Chi-square test
    chi2_stat, p_val = stats.chisquare(observed, expected_times)
    return chi2_stat, p_val

def chi_test2(study_quadrant_time, test_quadrant_time, total_time=None):
    """Perform Chi-square test to compare observed quadrant times between study and test phases."""
    if total_time is None:
        total_time = sum(study_quadrant_time.values()) + sum(test_quadrant_time.values())

    # Calculate expected time per quadrant (proportional to the total observed time)
    expected_time_per_quadrant = total_time / 8  # 8 values (4 quadrants * 2 phases)
    expected_times = [expected_time_per_quadrant] * 8  # 4 quadrants with equal expected time for each phase
    
    observed_study = [study_quadrant_time['UL'], study_quadrant_time['UR'], study_quadrant_time['LL'], study_quadrant_time['LR']]
    observed_test = [test_quadrant_time['UL'], test_quadrant_time['UR'], test_quadrant_time['LL'], test_quadrant_time['LR']]

    # Combine observed quadrant times for study and test phases
    observed = observed_study + observed_test
    
    # Perform the Chi-square test
    chi2_stat, p_val = stats.chisquare(observed, expected_times)
    return chi2_stat, p_val

def heatmaps(path_to_test, ending, hab):
    """Path to DLC folder to analyze place preference across both study and test phases."""
    animal = []
    proced = []
    
    # Load object positions from the group file
    object_pos = pd.read_excel(((os.path.dirname(path_to_test))) + '/object_pos_group.xlsx', index_col=0)

    # Extract animal IDs and procedures
    for filename in os.listdir(path_to_test+'/DLCJulia'): 

        if filename.endswith(ending):
            animal.append(filename.split('_')[0] + '_')
            p = filename.split('_')[1]
            proced.append(p.split('D')[0])

    # Iterate over each animal
    for idx, ani in enumerate(list(set(animal))):
        print(f'Analyzing {ani} {proced[0]} {proced[1]}')
        
        # Load data for both study and test phases
        df_study = process_df(path_to_test+'/DLCJulia/' + ani + proced[0] + ending)
        df_test = process_df(path_to_test+'/DLCJulia/' + ani + proced[1] + ending)
      
        # Combine both study and test phase data
        combined_df = pd.concat([df_study, df_test])
        
        # Get dynamic limits of the heatmap
        x_min, x_max = combined_df['TailBase']['x'].min(), combined_df['TailBase']['x'].max()
        y_min, y_max = combined_df['TailBase']['y'].min(), combined_df['TailBase']['y'].max()

        # Calculate the midpoints for each of the 4 quadrants
        x_mid = (x_max + x_min) / 2
        y_mid = (y_max + y_min) / 2

        
        
        # Calculate time spent in each quadrant for both study and test phases
        quadrant_time_study = calculate_quadrant_time(df_study, x_mid, y_mid)
        quadrant_time_test = calculate_quadrant_time(df_test, x_mid, y_mid)
        # Perform Chi-square test for study and test phases independently
        
        
        """chi2_study, p_val_study = chi_square_test(quadrant_time_study)
        chi2_test, p_val_test = chi_square_test(quadrant_time_test)
        if p_val_study > 0.05:
            print (f"Study phase NOT significantChi-square: {chi2_study}, p-value: {p_val_study}")
        elif p_val_test > 0.05:
            print (f"Test phase NOT significantChi-square: {chi2_study}, p-value: {p_val_study}")
        else:
            print(f"Study phase Chi-square: {chi2_study}, p-value: {p_val_study}")
            print(f"Test phase Chi-square: {chi2_test}, p-value: {p_val_test}")
        # Plotting
        """
        
        chi2_stat, p_val = chi_test2(quadrant_time_study, quadrant_time_test)
        
        print(f"Study vs Test Chi-square: {chi2_stat}, p-value: {p_val}")
        if p_val > 0.05:
            print(f"{ani} shows place preference!")
        
        
        
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        # Study Phase
        ax1.bar(quadrant_time_study.keys(), quadrant_time_study.values(), alpha=0.7)
        ax1.set_title(f'{ani} Study Phase Quadrant Times')
        ax1.set_ylabel('Frames')
        ax1.set_xlabel('Quadrant')

        # Test Phase
        ax2.bar(quadrant_time_test.keys(), quadrant_time_test.values(), alpha=0.7)
        ax2.set_title(f'{ani} Test Phase Quadrant Times')
        ax2.set_ylabel('Frames')
        ax2.set_xlabel('Quadrant')
        
        plt.tight_layout()
        if os.path.exists(os.path.dirname(path_to_test)+'/heatmaps'):
            print('saving:', )
            plt.savefig(os.path.dirname(path_to_test)+'/heatmaps/'+ani+proced[1]+'_hist.png')
        else:
            print((os.path.dirname(path_to_test))+'/heatmaps')
            os.makedirs(os.path.dirname(path_to_test)+'/heatmaps')
            plt.savefig(os.path.dirname(path_to_test)+'/heatmaps/'+ani+proced[1]+'_hist.png')
        plt.show()
        
        # Plot heatmaps of the study and test phases side by side
        heatmap_study, xedges, yedges = np.histogram2d(df_study['TailBase']['x'].dropna(), df_study['TailBase']['y'].dropna(), bins=15)
        heatmap_test, xedges1, yedges1 = np.histogram2d(df_test['TailBase']['x'].dropna(), df_test['TailBase']['y'].dropna(), bins=15)

        # Set the extent for both heatmaps to make them comparable
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        extent1 = [xedges1[0], xedges1[-1], yedges1[0], yedges1[-1]]
        # Compute a global color scale based on both heatmaps
        vmin = min(heatmap_study.min(), heatmap_test.min())
        vmax = max(heatmap_study.max(), heatmap_test.max())
        
        # Create a figure to hold both subplots
        plt.clf()
        f = plt.figure(figsize=(9.5, 3))
        ax = f.add_subplot(121)  # First subplot (study phase)
        ax2 = f.add_subplot(122)  # Second subplot (test phase)

       # Plot the study phase heatmap
        im = ax.imshow(heatmap_study.T, extent=extent, origin='lower', vmin=vmin, vmax=vmax)
        group = object_pos.group.loc[(object_pos.Animal_ID == ani.split('_')[0])].values[0]
        
        title_color = 'orange' if group == 'Chrim' else 'black'
        if hab == False:
            # Ensure that quadrant boundaries fit within the extent of the heatmap
            ax.plot([extent[0], x_mid], [extent[3], extent[3]])  # Upper Left boundary (horizontal)
            ax.plot([extent[0], x_mid], [extent[2], extent[2]])  # Lower Left boundary (horizontal)
            ax.plot([x_mid, extent[1]], [extent[3], extent[3]])  # Upper Right boundary (horizontal)
            ax.plot([x_mid, extent[1]], [extent[2], extent[2]])  # Lower Right boundary (horizontal)
            
            ax.plot([extent[1], extent[1]], [extent[2], extent[3]])  # Right boundary (vertical)
            ax.plot([extent[0], extent[0]], [extent[2], extent[3]])  # Left boundary (vertical)
            ax.plot([x_mid, x_mid], [extent[2], extent[3]])  # Vertical line splitting left and right
            ax.plot([extent[0], extent[1]], [y_mid, y_mid])  # Horizontal line splitting top and bottom

            # Retrieve object positions specific to the study phase
            ul_obj = object_pos.UL.loc[(object_pos.Animal_ID == ani.split('_')[0]) & (object_pos.Test == proced[0])].values[0]
            ur_obj = object_pos.UR.loc[(object_pos.Animal_ID == ani.split('_')[0]) & (object_pos.Test == proced[0])].values[0]
            ll_obj = object_pos.LL.loc[(object_pos.Animal_ID == ani.split('_')[0]) & (object_pos.Test == proced[0])].values[0]
            lr_obj = object_pos.LR.loc[(object_pos.Animal_ID == ani.split('_')[0]) & (object_pos.Test == proced[0])].values[0]

            # Plot object positions for the study phase
            l_mid= (extent[1]-extent[0])/4+extent[0]
            r_mid= (extent[1]-extent[0])*3/4+extent[0]
            low_mid= (extent[3]-extent[2])/4+extent[2]
            up_mid= (extent[3]-extent[2])*3/4+extent[2]
            ax.text(l_mid-20, up_mid, 'O'+str(ul_obj), fontsize='medium', color='red')  # Midpoint of UL quadrant
            ax.text(r_mid-20, up_mid, 'O'+str(ur_obj), fontsize='medium', color='red')  # Midpoint of UR quadrant
            ax.text(l_mid-10, low_mid, 'O'+str(ll_obj), fontsize='medium', color='red')  # Midpoint of LL quadrant
            ax.text(r_mid-10, low_mid, 'O'+str(lr_obj), fontsize='medium', color='red')  # Midpoint of LR quadrant
            
            ax.set_title(f'{ani} {proced[0]}', color=title_color)
       

        # Add colorbar to the study heatmap
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("", rotation=-90, va="bottom")

        # Plot the test phase heatmap
        im = ax2.imshow(heatmap_test.T, extent=extent1, origin='lower', vmin=vmin, vmax=vmax)

        if hab == False:
            # Ensure that quadrant boundaries fit within the extent of the heatmap
            ax2.plot([extent1[0], x_mid], [extent1[3], extent1[3]],)  # Upper Left boundary (horizontal)
            ax2.plot([extent1[0], x_mid], [extent1[2], extent1[2]])  # Lower Left boundary (horizontal)
            ax2.plot([x_mid, extent1[1]], [extent1[3], extent1[3]])  # Upper Right boundary (horizontal)
            ax2.plot([x_mid, extent1[1]], [extent1[2], extent1[2]])  # Lower Right boundary (horizontal)
            
            ax2.plot([extent1[1], extent1[1]], [extent1[2], extent1[3]])  # Right boundary (vertical)
            ax2.plot([extent1[0], extent1[0]], [extent1[2], extent1[3]])  # Left boundary (vertical)
            ax2.plot([x_mid, x_mid], [extent1[2], extent1[3]])  # Vertical line splitting left and right
            ax2.plot([extent1[0], extent1[1]], [y_mid, y_mid])  # Horizontal line splitting top and bottom

            # Retrieve object positions specific to the test phase
            ul_obj = object_pos.UL.loc[(object_pos.Animal_ID == ani.split('_')[0]) & (object_pos.Test == proced[1])].values[0]
            ur_obj = object_pos.UR.loc[(object_pos.Animal_ID == ani.split('_')[0]) & (object_pos.Test == proced[1])].values[0]
            ll_obj = object_pos.LL.loc[(object_pos.Animal_ID == ani.split('_')[0]) & (object_pos.Test == proced[1])].values[0]
            lr_obj = object_pos.LR.loc[(object_pos.Animal_ID == ani.split('_')[0]) & (object_pos.Test == proced[1])].values[0]

            # Plot object positions for the test phase
            l_mid= (extent1[1]-extent1[0])/4+extent1[0]
            r_mid= (extent1[1]-extent1[0])*3/4+extent1[0]
            low_mid= (extent1[3]-extent1[2])/4+extent1[2]
            up_mid= (extent1[3]-extent1[2])*3/4+extent1[2]
            ax2.text(l_mid-20, up_mid, 'O'+str(ul_obj), fontsize='medium', color='red')  # Midpoint of UL quadrant
            ax2.text(r_mid-20, up_mid, 'O'+str(ur_obj), fontsize='medium', color='red')  # Midpoint of UR quadrant
            ax2.text(l_mid-10, low_mid, 'O'+str(ll_obj), fontsize='medium', color='red')  # Midpoint of LL quadrant
            ax2.text(r_mid-10, low_mid, 'O'+str(lr_obj), fontsize='medium', color='red')  # Midpoint of LR quadrant
            

            ax2.set_title(f'{ani} {proced[1]}', color=title_color)

        # Add colorbar to the test heatmap
        cbar = ax2.figure.colorbar(im, ax=ax2)
        cbar.ax.set_ylabel("",rotation=-90, va="bottom")
        
        if os.path.exists(os.path.dirname(path_to_test)+'/heatmaps'):
            print('Saving at:', os.path.dirname(path_to_test)+'/heatmaps')
            plt.savefig(os.path.dirname(path_to_test)+'/heatmaps/'+ani+proced[1]+'.png')
        else:
            print('Creating:',os.path.dirname(path_to_test)+'/heatmaps')
            os.makedirs(os.path.dirname(path_to_test)+'/heatmaps')
            plt.savefig(os.path.dirname(path_to_test)+'/heatmaps/'+ani+proced[1]+'.png')
        plt.show()

ending= 'DLC_resnet50_JULIA_DLCJul25shuffle1_940000.csv'

DLC_fiber= 'DLC_resnet50_OIPFiberNetDec30shuffle1_105000.csv'
