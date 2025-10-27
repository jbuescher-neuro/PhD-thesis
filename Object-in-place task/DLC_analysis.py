# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 13:17:25 2025

@author: julch
"""

import cv2
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
import scipy.stats as stats
from statsmodels.stats.anova import AnovaRM
import pingouin as pg
#%%First load excel

def process_df(filename):
  data= pd.read_csv(filename, header=[1,2])
  
  #to avoid settings with copy warning
  data_Nose = data['Nose'].copy()
  data_LeftFront = data['FiberLeft'].copy()
  data_RightFront = data['FiberRight'].copy()
  data_Noseup = data["NoseFront"].copy()
  
  
  #drop less likely data
  data_Nose.drop(data_Nose[data_Nose['likelihood']<=0.9].index, inplace= True)
  
  data_LeftFront.drop(data_LeftFront[data_LeftFront['likelihood']<=0.9].index, inplace= True)

  data_RightFront.drop(data_RightFront[data_RightFront['likelihood']<=0.9].index, inplace= True)
 
  data_Noseup.drop(data_Noseup[data_Noseup['likelihood']<=0.9].index, inplace= True)
  
  #put it back to dataframe
  data['Nose']=data_Nose
  data['FiberRight']= data_RightFront
  #data['FiberRight']= data_RightFront
  data['FiberLeft']= data_LeftFront
  #data['ShoulderLeft']= data_LeftHind
  data['NoseFront']= data_Noseup

  return(data)
"""
def distance(df, ob_num, idx, x, y):
  obj= 'Object'+str(ob_num)

  dist= abs(((df[obj, 'x'][idx]-x)**2 + (df[obj, 'y'][idx]-y)**2)**0.5)
  return(dist)"""
def dist(df, bodypart, idx, object_points, ob_num):
  obj= 'Object'+str(ob_num)
  x= df[str(bodypart), 'x'][idx]
  y= df[str(bodypart), 'y'][idx]
  mid_x= object_points['xmid'][obj]
  mid_y= object_points['ymid'][obj]
  
  dist= abs(((mid_x-x)**2 + (mid_y-y)**2))**0.5
  return(dist)

"""Load video to define ROIs directly around object and take object exploration when nose is in this area 
(next step nose in area and close to mid than ears)"""


def nose_inROI(path, animal, proced, ending):
    frames_Object1=0
    frames_Object2=0
    frames_Object3=0
    frames_Object4=0
    exp_frames1= []
    exp_frames2= []
    exp_frames3= []
    exp_frames4= []
    fps=30
    freeze=0
    df_min= pd.DataFrame(index= ['Object1_min1', 'Object2_min1', 'Object3_min1', 'Object4_min1', 'Object1_min2', 'Object2_min2', 'Object3_min2', 'Object4_min2',
                                 'Object1_min3', 'Object2_min3', 'Object3_min3', 'Object4_min3', 'Object1_min4', 'Object2_min4', 'Object3_min4', 'Object4_min4',
                                 'Object1_min5', 'Object2_min5', 'Object3_min5', 'Object4_min5', 'Object1_min6', 'Object2_min6', 'Object3_min6', 'Object4_min6'])
                
    
    ROIs=[[],[],[],[]]
    objects= [[],[],[],[]]
    videopath= path+animal+proced+ending.split('.')[0]+'_labeled.mp4'
    len_x=375
    len_y= 365
    x_pixcm=11#len_x/31.15
    y_pixcm= len_y/31.15
    #get dataframe with mouse positions
    df= process_df(path+animal+proced+ending)
 
    
    
    #define Object ROIs
    if os.path.exists(path+'object_locations.csv'):
        obj= pd.read_csv(path+'object_locations.csv', usecols= [animal+proced])
        #converting list back into list of tinteger
        for idx, o in obj.iterrows():
            objects[idx]= o.values[0].split(',')
            objects[idx][0]= objects[idx][0].split('[')[1]
            objects[idx][-1]= objects[idx][-1].split(']')[0]
            objects[idx]= list(map(float, objects[idx]))
            objects[idx].extend([objects[idx][0]+objects[idx][2]/2, objects[idx][1]+objects[idx][3]/2])
        print("Object points:", objects)
    else:
        print(videopath)
        cap= cv2.VideoCapture(videopath)
        if (cap.isOpened()== False): 
          print("Error opening video stream or file")
        ret, frame = cap.read()
        ROIs= cv2.selectROIs('Select objects in the correct order. Press ENTER or SPACE after each ROI, then ESC to finish', frame)
        cv2.destroyAllWindows()
        
        dist2obj= 2
        for idx,ob in enumerate(ROIs):
            #print(ob)
            xmid= ((int(ob[2]))/2)+ob[0]
            ymid= ((int(ob[3]))/2)+ob[1]
            objects[idx]= [int(ob[0]-dist2obj *x_pixcm), int(ob[1]-dist2obj *y_pixcm), int(ob[0]+ob[2]+ dist2obj*x_pixcm), int(ob[1]+ob[3]+dist2obj*y_pixcm), xmid, ymid]
    object_points= pd.DataFrame(objects, ['Object1', 'Object2', 'Object3', 'Object4'], ['x1', 'y1', 'x2', 'y2', 'xmid', 'ymid'])
    #find min and max for x an y ranges:
    x_max= int(max(object_points['x2']))
    x_min= int(min(object_points['x1']))
    y_max= int(max(object_points['y2']))
    y_min= int(min(object_points['y1']))
   
    
    #loop through frames if Nose is recognized
    for idx,x in enumerate(df['Nose', 'x']):   
      
        #no Nose
        if np.isnan(x) == True:
            drop=+1
        elif np.isnan(df['NoseFront', 'x'][idx])== False:
            drop=+1
        else:
            #check if Nose is in range of any object
            if int(df['Nose', 'x'][idx]) in range(x_min, x_max+1) and int(df['Nose', 'y'][idx]) in range(y_min, y_max+1):
                #check if Nose within region of object for every object
                #check if nose was not moving more than x pixles the last 0,33 seconds
                if abs(df['Nose', 'x'][idx-10:idx+1].diff().sum())<5 and abs(df['Nose', 'y'][idx-10:idx+1].diff().sum())<5:
                    freeze+= 1
                    continue
                
                if object_points['x1']['Object1']< df['Nose','x'][idx]< object_points['x2']['Object1'] and object_points['y1']['Object1'] < df['Nose','y'][idx] < object_points['y2']['Object1']:
                    #if dist(df, 'Nose', idx, object_points, 1)< dist(df, 'EarLeft', idx, object_points, 1) and dist(df, 'Nose', idx, object_points, 1)< dist(df, 'EarRight', idx, object_points, 1):
                       frames_Object1+=1
                       exp_frames1.append(idx)
                
                elif object_points['x1']['Object2']< df['Nose','x'][idx]< object_points['x2']['Object2'] and object_points['y1']['Object2'] < df['Nose','y'][idx] < object_points['y2']['Object2']:
                  # if dist(df, 'Nose', idx, object_points, 2)< dist(df, 'EarLeft', idx, object_points, 2) and dist(df, 'Nose', idx, object_points, 2)< dist(df, 'EarRight', idx, object_points, 2):
                      frames_Object2+=1
                      exp_frames2.append(idx)
                      
                   
                elif object_points['x1']['Object3']< df['Nose','x'][idx]< object_points['x2']['Object3'] and object_points['y1']['Object3'] < df['Nose','y'][idx] < object_points['y2']['Object3']:
                    #if dist(df, 'Nose', idx, object_points, 3)< dist(df, 'EarLeft', idx, object_points, 3) and dist(df, 'Nose', idx, object_points, 3)< dist(df, 'EarRight', idx, object_points, 3):
                     frames_Object3+=1
                     exp_frames3.append(idx)
                
                elif object_points['x1']['Object4']< df['Nose','x'][idx]< object_points['x2']['Object4'] and object_points['y1']['Object4'] < df['Nose','y'][idx] < object_points['y2']['Object4']:
                    #if dist(df, 'Nose', idx, object_points, 4)< dist(df, 'EarLeft', idx, object_points, 4) and dist(df, 'Nose', idx, object_points, 4)< dist(df, 'EarRight', idx, object_points, 4):
                        frames_Object4+= 1
                        exp_frames4.append(idx)
                            
                        
    
    exp_frames= [exp_frames1, exp_frames2, exp_frames3, exp_frames4]
    
    expl_time= []
    #extract exploration times that are too short
    for o in exp_frames:
        c=0
        for idx, x in enumerate(o):
            if idx>0:
                #check if bout distance at least 4 frames
                #print(x-o[idx-1])
                if x-o[idx-1]==1:
                    c+=1
                else: #if new bout check how long bout was kick out short times (5 frames)
                    #print(np.shape(o))    
                    if c<6:
                        print('kick out!', c, o[idx-c:idx+1])
                        del o[idx-c:idx+1]
                    else:
                        expl_time.append(c/30)
                    
    #print(min(expl_time))
    
    expl_df= pd.DataFrame(exp_frames, index= ['Object1', 'Object2', 'Object3', 'Object4'])
    
    #get out short exploration times (0.1 sec 3 frames)
    """c=0
    for obj in range(0,4):
        for id, x in enumerate(expl_df[expl_df.columns[obj]]):
            if (id-1)>0:
                if x -expl_df[expl_df.columns[obj]][id-1]< 3:
                    c+=1 
                else:
                    if c<3:
                        expl_df[expl_df.columns[obj]]=expl_df[expl_df.columns[obj]][id-c:id]"""
    
    
    expl_df= expl_df.transpose()
    
    """disc ratio per minute"""
   
    for minu in range(1,7):
        df_min.loc[['Object1_min'+str(minu),'Object2_min'+str(minu),'Object3_min'+str(minu),'Object4_min'+str(minu)], animal+proced]=pd.Series([expl_df.Object1[expl_df.Object1<fps*minu*60].count()/fps, expl_df.Object2[expl_df.Object2<fps*minu*60].count()/fps, 
                                                                                                                                 expl_df.Object3[expl_df.Object3<fps*minu*60].count()/fps, expl_df.Object4[expl_df.Object4<fps*minu*60].count()/fps],                                                                                                         
                                                                                                                                 index=['Object1_min'+str(minu),'Object2_min'+str(minu),'Object3_min'+str(minu),'Object4_min'+str(minu)])
        """alt
        O3_4= (expl_df.Object3[expl_df.Object3<fps*minu*60].count()/fps)+(expl_df.Object4[expl_df.Object4<fps*minu*60].count()/fps)
        O1_2= (expl_df.Object1[expl_df.Object1<fps*minu*60].count()/fps)+(expl_df.Object2[expl_df.Object2<fps*minu*60].count()/fps)
        disc_min.loc['min'+str(minu)]= (O3_4-O1_2)/(O3_4+O1_2)"""
    
    for idx,i in enumerate(range(0,24,4)):
        df_min.loc['disc_ratio_min'+str(idx+1)]= df_min.apply(lambda row: ((row[i+2]+row[i+3])-(row[i+0]+row[i+1]))/(row[i+0]+row[i+1]+row[i+2]+row[i+3]),axis=0)
    
    
    expl_df.to_csv(os.path.dirname(os.path.dirname(path))+'/OpenCV/Expl/expl_frames_'+animal+proced+ending, index=False)
    
    time_object= [expl_df['Object1'].count()/30, expl_df['Object2'].count()/30, expl_df['Object3'].count()/30, expl_df['Object4'].count()/30]
    number_frames= df.shape[0]
    #print(x_min, x_max, y_min, y_max)
    #print('Number of frames without Nose:', drop, 'Number of frames', number_frames)
    print('freezing time', freeze/fps)
    return(expl_df, time_object, df_min, object_points)

#%%Analysis per test. videos from one test in one folder

def OIP_analysis(path_to_test, ending):
    """Performs automated analysis of Object-in-Place (OIP) task videos.
    For each animal and condition, it loads tracking data, quantifies time spent 
    and number of visits to each object, computes discrimination ratios, and saves results
    (exploration times, visit counts, object coordinates, and discrimination per minute) as 
    CSV and Pickle files in the specified folder."""
    time_object= []
    proced= []
    #mean_visit= []
    bouts= []
    animal= []
    proced= []
    objects= []
    
    
    #get animal and procedure names
    for filename in os.listdir(path_to_test):

        if filename.endswith(ending):
            
            #videoname=filename.split('.')[0]+'_labeled.mp4'
            #print(path+videoname)
            animal.append(filename.split('_')[0]+'_')
            p=filename.split('_')[1]
            proced.append(p.split('D')[0])
            #print(filename.split('_')[0]+'_')
    #create dataframes    
    df2= pd.DataFrame(index= ['Object1', 'Object2', 'Object3','Object4'])#, columns=[animal proced])
    df3= pd.DataFrame(index= ['Visits_Object1', 'Visits_Object2', 'Visits_Object3', 'Visits_Object4'])
    #df4= pd.DataFrame(index= ['meantime_Object1', 'meantime_Object2', 'meantime_Object3','meantime_Object4'])
    disc_ratio= pd.DataFrame()
    disc_min= pd.DataFrame()
    objects= pd.DataFrame( index=['Object1', 'Object2', 'Object3', 'Object4'])
    #analyze every video
    for idx, ani in enumerate(animal):
        print('Analyzing ', ani +proced[idx])
        expl_df, timeperobject, discpermin, Object_points= nose_inROI(path_to_test, ani, proced[idx], ending)
        
        #save time spend at each object
        time_object.append(timeperobject)
        
        #Get a list with x1, y1, x2, y2 pos as list for every object and put it in df    
        objects[ani+ proced[idx]]=pd.Series([Object_points.loc['Object1'][0:4].values.flatten().tolist(),Object_points.loc['Object2'][0:4].values.flatten().tolist(), 
                          Object_points.loc['Object3'][0:4].values.flatten().tolist(), Object_points.loc['Object4'][0:4].values.flatten().tolist()], 
                         index=['Object1', 'Object2', 'Object3', 'Object4'])
        
        disc_min[ani+ proced[idx]]= discpermin
        
        #Get object visits. Define as mouse wasnt at object for at least 3 frames could also go in nosein ROI function
        bout= [0, 0, 0, 0]
        
        for obj in range(0,4):
            for id, x in enumerate(expl_df[expl_df.columns[obj]]):
                if (id-1)>0:
                    if x -expl_df[expl_df.columns[obj]][id-1]> 3:
                        bout[obj]+=1
                    #elif x -expl_df[expl_df.columns[obj]][id-1]>1 and x -expl_df[expl_df.columns[obj]][id-1]<4: #distance between bouts
                        #print('too short', x -expl_df[expl_df.columns[obj]][id-1])
                        
        bouts.append(bout)
        #print(timeperobject, idx)
        
        df2[ani+proced[idx]]= pd.Series(time_object[idx], index=['Object1', 'Object2', 'Object3', 'Object4'])
         
        df3[ani+ proced[idx]]=pd.Series(bouts[idx], index=['Visits_Object1', 'Visits_Object2', 'Visits_Object3', 'Visits_Object4'])

        
        print(df2)
    
    
       
    #get the total exploration time 

    df2.loc['Total exploration']= df2.sum()
    df2.loc['disc_ratio']= df2.apply(lambda row: ((row[2]+row[3])-(row[0]+row[1]))/(row[0]+row[1]+row[2]+row[3]),axis=0)

    df2= pd.concat([df2, df3])
    
    disc_min.to_csv(path_to_test+ 'df_min.csv')
    df2.to_csv(path_to_test+'OIP_analysis.csv')
    objects.to_csv(path_to_test+'object_locations.csv')
    objects.to_pickle(path_to_test+'object_locations.pkl')
    return(df2, disc_min, objects)

#%% Track length also in clean_analysis
ending= 'DLC_resnet50_JULIA_DLCJul25shuffle1_940000.csv'

def track_length(path_to_test, ending):
    animal=[]
    proced= []
    px_cm=11
    
    #get animal and procedure names
    for filename in os.listdir(path_to_test): 
        if filename.endswith(ending):
            animal.append(filename.split('_')[0]+'_')
            p=filename.split('_')[1]
            proced.append(p.split('D')[0])
    print(animal)
    trav_dist= pd.DataFrame(index=np.arange(len(animal)),columns=['MouseID', 'Task', 'dist_min1', 'dist_min2', 'dist_min3', 'dist_min4', 'dist_min5', 'dist_min6'])
    for idx, ani in enumerate(animal):
        print('Analyzing ', ani +proced[idx])
        dist_trav= trav_dist.iloc[idx]
        dist_trav['MouseID']= ani
        dist_trav['Task']= proced[idx]
        df=process_df(path_to_test+ani+proced[idx]+ending)
        dist = [0]*6
        
        for idx,x in enumerate(df['FiberLeft', 'x'].dropna()):
            #idx= df.index[df['FiberLeft','x']==x][0]
            if idx>0:
                if np.isnan(df['FiberLeft', 'x'][idx]) or np.isnan(df['FiberLeft', 'x'][idx-1]):
                    continue
                else:
    
                    if idx< 6*30*60:
                        y=df['FiberLeft', 'y'].dropna()[idx]
                        x1= df['FiberLeft', 'x'][idx-1]
                        y1= df['FiberLeft', 'y'][idx-1]
                    
                    if (abs((x1-x)**2+(y1-y)**2)**0.5)/11 > 5:
                        continue
                    elif idx < 1*30*60: #per min video 30 fps
                        #print(dist[0],(abs((x1-x)**2+(y1-y)**2)**0.5)/11)
                        dist[0]= dist[0]+ (abs((x1-x)**2+(y1-y)**2)**0.5)/11
                    elif 1*30*60 < idx < 2*30*60:
                        dist[1]= dist[1]+ ((abs((x1-x)**2+(y1-y)**2)**0.5))/11
                    elif 2*30*60 < idx < 3*30*60:
                        dist[2]= dist[2]+ (abs((x1-x)**2+(y1-y)**2)**0.5)/11
                    elif 3*30*60 < idx < 4*30*60:
                        dist[3]= dist[3]+ (abs((x1-x)**2+(y1-y)**2)**0.5)/11
                    elif 4*30*60 < idx < 5*30*60:
                        dist[4]= dist[4]+ (abs((x1-x)**2+(y1-y)**2)**0.5)/11
                    elif 5*30*60 < idx < 6*30*60-5:
                        dist[5]= dist[5]+ (abs((x1-x)**2+(y1-y)**2)**0.5)/11
                    elif idx > 6*30*60:
                        print('> 6 min')
                        break
                dist_trav[['dist_min1', 'dist_min2', 'dist_min3', 'dist_min4', 'dist_min5', 'dist_min6']]= dist
    print(trav_dist.Task)
    trav_dist['sum']=trav_dist.loc[:,'dist_min1':'dist_min6'].T.sum()
    sns.stripplot(data=trav_dist, x= 'MouseID', y= 'sum', hue='Task')
    plt.xticks(rotation=45)
    plt.figure()
    plt.plot(range(1,7),trav_dist.loc[:, trav_dist.columns.str.contains('min')].T, label=trav_dist.MouseID)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid()
    plt.xlabel('minute')
    plt.ylabel('distance traveled in [cm]')
    plt.savefig(path_to_test+'dist_permin',  bbox_inches= 'tight')
    
    path_cohort=os.path.dirname(os.path.dirname((os.path.dirname(path_to_test))))
    print(path_cohort)
    if os.path.exists(path_cohort+'/track.csv'):
        print('track.csv is updated')
        track= pd.read_csv(path_cohort+'/track.csv', index_col=[0])
        track= pd.concat([track, trav_dist.loc[:,['MouseID','Task', 'sum']]], ignore_index= True)
        print(track)
    else:
        print('track.csv is created in '+ path_cohort+'/track.csv')
        track= trav_dist.loc[:,['MouseID','Task', 'sum']]
    track.to_csv(path_cohort+'/track1.csv')
    return(trav_dist,df)

def plot_dist_trav(path_cohort):
    track= pd.read_csv(path_cohort+'track.csv', index_col=[0])
    df_min= pd.read_csv(path_cohort+'df_min.csv', index_col=[0])
    track['group']=pd.Series(dtype='string')
    df_min['track']= pd.Series()
    
    for mouse in track.MouseID.unique():
       
        track['group'].loc[track.MouseID==mouse]= df_min.group.loc[df_min.MouseID==mouse.split('_')[0]].iloc[0]
        for task in df_min.Task.unique():
            print(mouse, task)
            df_min['track'].loc[(df_min.MouseID==mouse.split('_')[0]) & (df_min.Task==task)]= track['sum'].loc[(track.MouseID==mouse) & (track.Task == task)].iloc[0]
            #print(df_min['track'].loc[(df_min.MouseID==mouse.split('_')[0]) & (df_min.Task==task)])
            #print(track['sum'].loc[(track.MouseID==mouse) & (track.Task == task)])
    """Boxplots over tests"""
    plt.figure(figsize=(4,3.54), dpi=300)
    ax= plt.subplot()
    sns.boxplot(data=track[~track.Task.str.contains('Hab')], y= 'sum', x='Task', hue= 'group', hue_order=['Control', 'Chrim'], palette= {'Control':'dimgray','Chrim':'darkorange'},  order= sorted(track.Task[~track.Task.str.contains('Hab')].unique()), saturation=1, boxprops=dict(alpha=.45), width=0.4)
    sns.stripplot(data=track[~track.Task.str.contains('Hab')], y= 'sum', x='Task', hue= 'group', hue_order=['Control', 'Chrim'], order= sorted(track.Task[~track.Task.str.contains('Hab')].unique()),
                  palette= ['dimgrey','darkorange'], dodge=True, ax=ax,size=5.5, alpha=.8, edgecolor='k', linewidth=1, legend= False)
    
    plt.ylabel('track length')
    #plt.xticks(rotation=45, ha= 'right')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig(path_cohort+'track_test.svg')
    
    #Habituation plot
    plt.figure(figsize=(5,3.54), dpi=300)
    ax= plt.subplot()
    sns.boxplot(data=track[track.Task.str.contains('Hab')], y= 'sum', x='Task', hue= 'group', hue_order=['Control', 'Chrim'], palette= {'Control':'dimgray','Chrim':'darkorange'},  order= sorted(track.Task[track.Task.str.contains('Hab')].unique()), saturation=1, boxprops=dict(alpha=.45), width=0.4)
    sns.stripplot(data=track[track.Task.str.contains('Hab')], y= 'sum', x='Task', hue= 'group', hue_order=['Control', 'Chrim'], order= sorted(track.Task[track.Task.str.contains('Hab')].unique()),
                  palette= ['dimgrey','darkorange'], dodge=True, ax=ax,size=5.5, alpha=.8, edgecolor='k', linewidth=1, legend= False)
    
    plt.ylabel('track length')
    plt.xticks(rotation=45, ha='right')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig(path_cohort+'track_Hab.png')
    
    #disc ratio track length
    plt.figure()
    
    sns.stripplot(data=df_min[df_min.Task.str.contains('Test')], x='track', y='disc_ratio_min3', hue='group')
    
    aov= pg.mixed_anova(dv= 'sum', within='Task', between='group', subject= 'MouseID', data=track.loc[track.Task.str.contains('Study')])
    print('Repeated Anova for Study Phase')
    post= pg.pairwise_ttests(dv= 'sum', within='Task', between= 'group', subject='MouseID', data= track.loc[track.Task.str.contains('Study')])
    print(aov, post)
    
    aov= pg.mixed_anova(dv= 'sum', within='Task', between='group', subject= 'MouseID', data=track.loc[track.Task.str.contains('Test')])
    print('Repeated Anova for Test Phase')
    post= pg.pairwise_ttests(dv= 'sum', within='Task', between= 'group', subject='MouseID', data= track.loc[track.Task.str.contains('Test')])
    print(aov, post)
    
    aov= pg.mixed_anova(dv= 'sum', within='Task', between='group', subject= 'MouseID', data=track.loc[track.Task.str.contains('Hab')])
    print('Repeated Anova for Habituation')
    post= pg.pairwise_ttests(dv= 'sum', within='Task', between= 'group', subject='MouseID', data= track.loc[track.Task.str.contains('Hab')])
    print(aov,post)
    return(track,df_min)
"""heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

plt.clf()
plt.imshow(heatmap.T, extent=extent, origin='lower')
plt.show()"""

def heatmaps(path_to_test, ending, hab):
    
    """Path to DLC folder!"""
    animal=[]
    proced= []
    UL= [160, 340]
    UR= [350, 340]
    LL= [160, 100]
    LR= [350, 100]
    object_pos=pd.read_excel(os.path.dirname(os.path.dirname(os.path.dirname(path_to_test)))+'/object_pos_group.xlsx', index_col=0)
    
    for filename in os.listdir(path_to_test): 
        if filename.endswith(ending):
            animal.append(filename.split('_')[0]+'_')
            p=filename.split('_')[1]
            proced.append(p.split('D')[0])
    print(proced)
    
    for idx, ani in enumerate(list(set(animal))):
        print('Analyzing ', ani +proced[0] + proced[1]) 
        df=process_df(path_to_test+ani+proced[0]+ending)

        heatmap, xedges, yedges = np.histogram2d(df['FiberLeft','x'].dropna(), df['FiberLeft', 'y'].dropna(), bins= 15)
        extent= [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        
        df=process_df(path_to_test+ani+proced[1]+ending)
        heatmap1, xedges1, yedges1 = np.histogram2d(df['FiberLeft','x'].dropna(), df['FiberLeft', 'y'].dropna(), bins= 15)
        extent1= [xedges1[0], xedges1[-1], yedges1[0], yedges1[-1]]
        
        plt.clf()
        #plt.subplot(1,2, 1) 
        f = plt.figure(figsize=(9.5,3))
        ax = f.add_subplot(121)
        ax2 = f.add_subplot(122)
        #plt.figure(figsize=(9,9))
        #plt.xlim([0, 460])
        #plt.ylim([20,450])
        im=ax.imshow(heatmap.T, extent=extent, origin='lower')
        if hab== False:
            ax.text(UL[0],UL[1], 'O'+str(object_pos.UL.loc[(object_pos.Animal_ID==ani.split('_')[0]) & (object_pos.Test==proced[idx])][0]), fontsize= 'medium')
            ax.text(UR[0],UR[1], 'O'+str(object_pos.UR.loc[(object_pos.Animal_ID==ani.split('_')[0]) & (object_pos.Test==proced[idx])][0]), fontsize= 'medium')
            ax.text(LL[0],LL[1], 'O'+str(object_pos.LL.loc[(object_pos.Animal_ID==ani.split('_')[0]) & (object_pos.Test==proced[idx])][0]), fontsize= 'medium')
            ax.text(LR[0],LR[1], 'O'+str(object_pos.LR.loc[(object_pos.Animal_ID==ani.split('_')[0]) & (object_pos.Test==proced[idx])][0]), fontsize= 'medium')
        
        if object_pos.group.loc[(object_pos.Animal_ID==ani.split('_')[0])][0] == 'Chrim':
            ax.set_title(ani+ ' ' +proced[0], color='orange')
        else:
            ax.set_title(ani+ ' ' +proced[0], color='black')
        
        cbar = ax.figure.colorbar(im, ax = ax)
        cbar.ax.set_ylabel("Color bar", rotation = -90, va = "bottom")
        #plt.subplot(1,2, 2) #
        #plt.figure(figsize=(9,9))
        #plt.xlim([0, 500])
        #plt.ylim([0,450])
        im=ax2.imshow(heatmap1.T, extent=extent1, origin='lower')
        if hab== False:
            plt.text(UL[0],UL[1], 'O'+str(object_pos.UL.loc[(object_pos.Animal_ID==ani.split('_')[0]) & (object_pos.Test==proced[idx])][0]), fontsize= 'medium')
            plt.text(UR[0],UR[1], 'O'+str(object_pos.UR.loc[(object_pos.Animal_ID==ani.split('_')[0]) & (object_pos.Test==proced[idx])][0]), fontsize= 'medium')
            plt.text(LL[0],LL[1], 'O'+str(object_pos.LL.loc[(object_pos.Animal_ID==ani.split('_')[0]) & (object_pos.Test==proced[idx])][0]), fontsize= 'medium')
            plt.text(LR[0],LR[1], 'O'+str(object_pos.LR.loc[(object_pos.Animal_ID==ani.split('_')[0]) & (object_pos.Test==proced[idx])][0]), fontsize= 'medium')
        
        if object_pos.group.loc[(object_pos.Animal_ID==ani.split('_')[0])][0] == 'Chrim':
            plt.title(ani+ ' ' +proced[1], c='orange')
        else:
            plt.title(ani+ ' ' +proced[1], c='black')
        
        cbar = ax.figure.colorbar(im, ax = ax2)
        cbar.ax.set_ylabel("Color bar", rotation = -90, va = "bottom")
        
        if os.path.exists(os.path.dirname(os.path.dirname(path_to_test))+'/heatmaps'):
            plt.savefig(os.path.dirname(os.path.dirname(path_to_test))+'/heatmaps/'+ani+'.png')
        else:
            os.makedirs(os.path.dirname(os.path.dirname(path_to_test))+'/heatmaps')
            plt.savefig(os.path.dirname(os.path.dirname(path_to_test))+'/heatmaps/'+ani+'.png')
        plt.show()

#%% Plot object exploration per object (each study and test phase), ROI size (object_locations)

def obj_expl(path):
    #oip= pd.read_csv(path+ 'OIP_analysis.csv', nrows= 4, index_col=0)
    oip= pd.read_csv(path+ 'df_min_aut.csv',index_col=0)
    study= oip.loc[:, oip.columns.str.contains('Study')]
    test= oip.loc[:, oip.columns.str.contains('Test')]
    #sns.relplot(data= study )
    #sns.boxplot(x= study.index, data= study, orient= "Vertical")
    
    #study= study.T
    plt.figure()
   
    #for ani in study.columns:
        #sns.relplot(x= study.index[0:4], y= study[ani][0:4])
        #plt.scatter(study.index[0:4], study[ani][0:4])
        
    for idx,obj in enumerate(study.index[0:4]):
        plt.boxplot([study.loc[obj]], positions= [idx])
        #sns.boxplot(x= obj, y= study.loc[obj], )
    f,p= stats.f_oneway(study.loc['Object1_min6'], study.loc['Object2_min6'], study.loc['Object3_min6'], study.loc['Object4_min6'])
    plt.title('Exploration per Object in Study phase for '+str(path.split('/')[-2]))
    plt.xticks(ticks= [0, 1, 2, 3], labels= [1, 2,3,4])
    plt.xlabel('Object')
    plt.ylabel('Exploration time [s]')
    plt.text(-1,-7,'Anova p value= ' +str(p))
    plt.savefig(path+'expl_obj_study.png')
    plt.show()
   
    print('Anova p value for study= ', p)
    
    for idx,obj in enumerate(test.index[0:4]):
        plt.boxplot([test.loc[obj]], positions= [idx])
        
    f,p= stats.f_oneway(test.loc['Object1_min6'], test.loc['Object2_min6'],test.loc['Object3_min6'], test.loc['Object4_min6'])
    plt.title('Exploration per Object in Test phase for '+str(path.split('/')[-2]))
    plt.xticks(ticks= [0, 1, 2, 3], labels= [1, 2,3,4])
    plt.xlabel('Object')
    plt.ylabel('Exploration time [s]')
    plt.text(-1,-10,'Anova p value= ' +str(p))
    plt.savefig(path+'expl_obj_test.png')
    plt.show()
    
    print('Anova p value for test= ', p)
    
    #ROI size
    x_pxpercm= 11
    y_pxpercm= 365/31.15
    objects= pd.read_csv(path+'object_locations.csv', index_col=0) 
     
    print(objects)
    """just if list is a string which happens when saving as csv"""
    for col in objects:
        objects[col]=objects[col].apply(literal_eval)
        #for ob in objects.index:
            #objects[col].loc[ob]=objects[col].loc[ob].strip('[]').split(',')
        
    #get the exploration time dependent of ROI size
    study_o= objects.loc[:, objects.columns.str.contains('Study')]
    test_o = objects.loc[:, objects.columns.str.contains('Test')]
    colors= ['b', 'k', 'y', 'm']
    label= ['Object1', 'Object2', 'Object3', 'Object4']
    i=0
    
    
    for obj in study_o.index[0:4]:
        x_len=[]
        y_len=[]
        area= []
        
        for idx in range(0,len(study_o.columns)): 
            x_len.append(abs(float(study_o.loc[obj][idx][0])-float(study_o.loc[obj][idx][2]))/x_pxpercm)
            y_len.append(abs(float(study_o.loc[obj][idx][1])-float(study_o.loc[obj][idx][3]))/x_pxpercm)
            
            area.append(x_len[-1]*y_len[-1])
        
        print(i, obj)
        plt.scatter(area, study.loc[obj], color= colors[i], label= label[i]) 
        i+=1
        
    plt.xlabel('area of ROI [$cm^2$]')
    plt.ylabel('exploration time [s]')
    plt.title('exploration in dependence of ROI size in Study of '+str(path.split('/')[-2]))
    plt.legend()
    plt.savefig(path+ path.split('/')[-2]+'expl_ROI.png')
    
    
#%%Plot disc_ratio per minute

def plot_disc_min(path):
    
    df_min= pd.read_csv(path+'df_min.csv', index_col=0)
    #study= disc_min.loc[:,disc_min.columns.str.contains('Study')]
    test= df_min.loc[df_min.index.str.contains('disc'),df_min.columns.str.contains('Test')]
    #study= study.reindex(sorted(study.columns), axis=1)
    test= test.reindex(sorted(test.columns), axis=1)
    label= test.columns#[a[0] for a in test.columns.str.split('_')]
    
    #fig,ax= plt.subplot(1,1, sharex= True)
    for idx in range(0,test.columns.size):
        #if test.columns[idx]
        plt.plot(range(1,7), test.iloc[:,idx], '-', label=label[idx])
        #ax[0,1].plot(range(1,7), study.iloc[:,idx], '-', label=label[idx])
        
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('minute')
    plt.ylabel('disc ratio')
    plt.title('cumulative disc ratio per minute')
    plt.axhline(0, 0, color= 'k')
    plt.grid()
    plt.savefig(path+'cum_disc_ratio.png', bbox_inches= 'tight')
  
def bland_altman(path_man, path_aut):
    #Transpose manual dataframe and drop first row which should be coulmns
    df_min_man= pd.read_csv(path_man+'df_min.csv', header= 0, index_col=0)
    # Step 1: Create a MouseID_Task column in df_min_man
    df_min_man['MouseID_Task'] = df_min_man['MouseID'] + '_' + df_min_man['Task']

    # Step 2: Set index to MouseID_Task and transpose the relevant disc_ratio columns
    disc_cols = [col for col in df_min_man.columns if 'disc_ratio_min' in col]
    df_min_man = df_min_man.set_index('MouseID_Task').T
    
    # Now df_min_man_wide and df_min_aut are in the same shape: 
    # - rows: disc_ratio_min1 to disc_ratio_min6
    # - columns: MouseID_Task strings

        
    df_min_aut= pd.read_csv(path_aut+'df_min_aut.csv', header= 0, index_col=0)
    df_min_man= df_min_man.reindex(sorted(df_min_man.columns), axis=1)
    df_min_aut= df_min_aut.reindex(sorted(df_min_aut.columns), axis=1)
    
    
    # Step 0: Only keep rows matching real object exploration (Object1–4_min1–6), exclude "Visits_Object"
    object_rows = df_min_man.index[df_min_man.index.str.match(r'^Object\d+_min6')]
    man_df = df_min_man.loc[object_rows]
    aut_df = df_min_aut.loc[object_rows]
    print(man_df)
    # Step 1: Flatten to long format
    man_long = man_df.stack().reset_index()
    man_long.columns = ['Object_minute', 'Mouse_Session', 'Manual']
    
    aut_long = aut_df.stack().reset_index()
    aut_long.columns = ['Object_minute', 'Mouse_Session', 'Automated']
    
    # Step 2: Merge both
    merged = pd.merge(man_long, aut_long, on=['Object_minute', 'Mouse_Session'])
    
    # Step 3: Compute mean and difference
    merged['Mean'] = merged[['Manual', 'Automated']].mean(axis=1)
    merged['Diff'] = merged['Automated'] - merged['Manual']
    
    # Step 4: Bland-Altman Plot
    mean_diff = merged['Diff'].mean()
    std_diff = merged['Diff'].std()
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff
    
    plt.figure(figsize=(8,6))
    ax=plt.subplot()
    plt.scatter(merged['Mean'], merged['Diff'], alpha=0.6)
    plt.axhline(mean_diff, color='red', linestyle='--', label=f'Mean Diff = {mean_diff:.2f}')
    plt.axhline(loa_upper, color='gray', linestyle=':', label=f'+1.96 SD = {loa_upper:.2f}')
    plt.axhline(loa_lower, color='gray', linestyle=':', label=f'-1.96 SD = {loa_lower:.2f}')
    plt.xlabel('Mean Exploration Time (s)')
    plt.ylabel('Difference (Automated - Manual)')
    plt.title('Bland–Altman Plot: Object Exploration Times')
    plt.grid(False)
    plt.legend()
    plt.tight_layout()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(path_man+'BlandAltman.svg')
    plt.show()
  
    #disc_ratio
    # Only keep disc_ratio rows from both dataframes
    disc_rows = df_min_aut.index[df_min_aut.index.str.startswith('disc_ratio')]
    disc_man = df_min_man.loc[disc_rows]
    disc_aut = df_min_aut.loc[disc_rows]
    
    # Align and reshape
    disc_man_long = disc_man.stack().reset_index()
    disc_man_long.columns = ['disc_ratio', 'Mouse_Session', 'Manual']
    
    disc_aut_long = disc_aut.stack().reset_index()
    disc_aut_long.columns = ['disc_ratio', 'Mouse_Session', 'Automated']
    
    # Merge
    disc_merged = pd.merge(disc_man_long, disc_aut_long, on=['disc_ratio', 'Mouse_Session'])
    
    # Convert to numeric and drop rows with non-numeric entries
    disc_merged['Manual'] = pd.to_numeric(disc_merged['Manual'], errors='coerce')
    disc_merged['Automated'] = pd.to_numeric(disc_merged['Automated'], errors='coerce')
    
    # Drop rows with missing values (e.g., due to conversion failure)
    disc_merged = disc_merged.dropna(subset=['Manual', 'Automated'])
    
    # Now re-run the plot
    sns.lmplot(data=disc_merged, x='Manual', y='Automated', height=6, aspect=1)
    plt.title('Discrimination Ratio: Manual vs Automated')
    plt.tight_layout()
    plt.savefig(path_man+'corr_aut_man.svg')
    plt.show()
    
    # Pearson/Spearman correlation
    from scipy.stats import pearsonr, spearmanr
    
    pearson_corr, p_pearson = pearsonr(disc_merged['Manual'], disc_merged['Automated'])
    spearman_corr, p_spearman = spearmanr(disc_merged['Manual'], disc_merged['Automated'])
    
    print(f"Pearson r = {pearson_corr:.3f} (p = {p_pearson:.4f})")
    print(f"Spearman ρ = {spearman_corr:.3f} (p = {p_spearman:.4f})")

    
def diff_disc_min(path_man, path_aut):
    #rows_man= range(1,26)
    #rows_aut=
    #Transpose manual dataframe and drop first row which should be coulmns
    df_min_man= pd.read_csv(path_man+'df_min.csv', header= 0, index_col=0)
    # Step 1: Create a MouseID_Task column in df_min_man
    df_min_man['MouseID_Task'] = df_min_man['MouseID'] + '_' + df_min_man['Task']
    #print(df_min_man)
    # Step 2: Set index to MouseID_Task and transpose the relevant disc_ratio columns
    disc_cols = [col for col in df_min_man.columns if 'disc_ratio_min' in col]
    df_min_man_wide = df_min_man.set_index('MouseID_Task')[disc_cols].T
    
    # Now df_min_man_wide and df_min_aut are in the same shape: 
    # - rows: disc_ratio_min1 to disc_ratio_min6
    # - columns: MouseID_Task strings
    

        
    df_min_aut= pd.read_csv(path_aut+'df_min.csv', header= 0, index_col=0)
    df_min_man= df_min_man.reindex(sorted(df_min_man.columns), axis=1)
    df_min_aut= df_min_aut.reindex(sorted(df_min_aut.columns), axis=1)
    
    # Step 3: Align columns (intersection of common columns in both)
    common_cols = df_min_aut.columns.intersection(df_min_man_wide.columns)
    df_min_man = df_min_man_wide[common_cols].astype(float)
    
    """Bland-Altman Plot"""
    
    # 1. Extract only object exploration rows (exclude discrimination ratios)
    object_rows = df_min_man.index.str.startswith('Object')
    man_df = df_min_man[object_rows]
    aut_df = df_min_aut[object_rows]
    
    # 2. Flatten both DataFrames to long format for direct comparison
    man_long = man_df.stack().reset_index()
    man_long.columns = ['Object_minute', 'Mouse_Session', 'Manual']
    
    aut_long = aut_df.stack().reset_index()
    aut_long.columns = ['Object_minute', 'Mouse_Session', 'Automated']
    
    # 3. Merge both datasets on Object+Minute and Mouse_Session
    merged = pd.merge(man_long, aut_long, on=['Object_minute', 'Mouse_Session'])
    
    # 4. Compute means and differences
    merged['Mean'] = merged[['Manual', 'Automated']].mean(axis=1)
    merged['Diff'] = merged['Automated'] - merged['Manual']
    
    # 5. Bland-Altman plot
    mean_diff = merged['Diff'].mean()
    std_diff = merged['Diff'].std()
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff
    
    plt.figure(figsize=(8,6))
    plt.scatter(merged['Mean'], merged['Diff'], alpha=0.6)
    plt.axhline(mean_diff, color='red', linestyle='--', label=f'Mean Diff = {mean_diff:.2f}')
    plt.axhline(loa_upper, color='gray', linestyle=':', label=f'+1.96 SD = {loa_upper:.2f}')
    plt.axhline(loa_lower, color='gray', linestyle=':', label=f'-1.96 SD = {loa_lower:.2f}')
    plt.xlabel('Mean Exploration Time (s)')
    plt.ylabel('Difference (Automated - Manual)')
    plt.title('Bland–Altman Plot: Object Exploration Times')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    """diff disc ratio"""
    # Step 4: Subset both DataFrames to only common columns
    
    df_min_aut = df_min_aut.loc[df_min_aut.index.str.contains('disc_ratio_min'), common_cols].astype(float)
    #print(test_man)
    
    #convert to float 
    df_min_man.loc['disc_ratio_min1': 'disc_ratio_min6']= df_min_man.loc['disc_ratio_min1': 'disc_ratio_min6'].astype(float)
    df_min_aut.loc['disc_ratio_min1': 'disc_ratio_min6']= df_min_aut.loc['disc_ratio_min1': 'disc_ratio_min6'].astype(float)
   
    test_man= df_min_man.loc[df_min_man.index.str.contains('disc'),df_min_man.columns.str.contains('Test')]
    test_aut= df_min_aut.loc[df_min_aut.index.str.contains('disc'),df_min_aut.columns.str.contains('Test')]
    study_man= df_min_man.loc[df_min_man.index.str.contains('disc'),df_min_man.columns.str.contains('Study')]
    study_aut= df_min_aut.loc[df_min_aut.index.str.contains('disc'),df_min_aut.columns.str.contains('Study')]
    
    
    print(test_aut, 'test_aut')
    label= [a[0] for a in test_man.columns.str.split('_')]
    
    fig,ax= plt.subplots(2,1, sharex=True)
    for idx in range(0,test_man.columns.size):
        
        ax[1].plot(range(1,7), test_aut.iloc[:,idx]-test_man.iloc[:,idx], '-', label=label[idx])
        ax[1].set_title('Test phase')
        ax[0].set_title('Study phase')
        ax[0].plot(range(1,7), study_aut.iloc[:,idx]-study_man.iloc[:,idx], '-', label=label[idx])
        #print(study_aut.iloc[:,idx]-study_man.iloc[:,idx], study_aut.iloc[:,idx], study_man.iloc[:,idx])
        
    plt.legend(label,loc='center left', bbox_to_anchor=(1, 0.5))
    for a in ax.flat:
        a.set(label='minute',ylabel='Difference in disc', xlabel='minute')
        a.axhline(0, 0, color= 'k')
        a.grid()
    #plt.xlabel('minute')
    #plt.ylabel('disc ratio')
    fig.suptitle('difference in disc ratio per minute (Aut-JB)')
    #plt.axhline(0, 0, color= 'k')
    #plt.grid()
    plt.xticks([1,2,3,4,5,6])
    plt.tight_layout()
    plt.savefig(path_aut+'diff_disc_ratio.png', bbox_inches= 'tight')
    
    #
    
    
def diff_expl_min(path_man, path_aut):
    df_min_man= pd.read_csv(path_man+'df_min.csv', index_col=0)
    df_min_aut= pd.read_csv(path_aut+'df_min.csv', index_col=0)
    #convert to float
    df_min_man.loc['Object1_min1': 'Object4_min6']= df_min_man.loc['Object1_min1': 'Object4_min6'].astype(float)
    df_min_aut.loc['Object1_min1': 'Object4_min6']= df_min_aut.loc['Object1_min1': 'Object4_min6'].astype(float)
    """ accumulated error
    df_man_test= df_min_man.loc[df_min_man.index.str.contains('Object'),df_min_man.columns.str.contains('Test')]
    df_aut_test= df_min_aut.loc[df_min_aut.index.str.contains('Object'),df_min_aut.columns.str.contains('Test')]
    label_test= df_man_test.columns
    df_man_test= df_man_test.reindex(sorted(df_man_test.columns), axis=1)
    df_aut_test= df_aut_test.reindex(sorted(df_aut_test.columns), axis=1)
    
    df_man_study= df_min_man.loc[df_min_man.index.str.contains('Object'),df_min_man.columns.str.contains('Study')]
    df_aut_study= df_min_aut.loc[df_min_aut.index.str.contains('Object'),df_min_aut.columns.str.contains('Study')]
    label_study= df_man_study.columns
    df_man_study= df_man_study.reindex(sorted(df_man_study.columns), axis=1)
    df_aut_study= df_aut_study.reindex(sorted(df_aut_study.columns), axis=1)"""
    
    #create a dataframe with the errors per minute not accumulated
    dif_obj_man= pd.DataFrame(columns= df_min_man.columns)
    dif_obj_aut= pd.DataFrame(columns= df_min_aut.columns)
    for o in range(1,5):
        dif_obj_man= pd.concat([dif_obj_man, df_min_man.loc[df_min_man.index.str.contains('Object'+str(o))].diff()])
        dif_obj_aut= pd.concat([dif_obj_aut, df_min_aut.loc[df_min_aut.index.str.contains('Object'+str(o))].diff()])
    dif_obj_man.loc[dif_obj_man.index.str.contains('min1'),:]=df_min_man.loc[df_min_man.index.str.contains('min1'),:]
    dif_obj_aut.loc[dif_obj_man.index.str.contains('min1'),:]=df_min_aut.loc[df_min_aut.index.str.contains('min1'),:]
    
    df_man_test= dif_obj_man.loc[:,dif_obj_man.columns.str.contains('Test')]
    df_aut_test= dif_obj_aut.loc[:,dif_obj_aut.columns.str.contains('Test')]
    df_man_test= df_man_test.reindex(sorted(df_man_test.columns), axis=1)
    df_aut_test= df_aut_test.reindex(sorted(df_aut_test.columns), axis=1)
   
    
    df_man_study= dif_obj_man.loc[:,dif_obj_man.columns.str.contains('Study')]
    df_aut_study= dif_obj_aut.loc[:,dif_obj_aut.columns.str.contains('Study')]
    df_man_study= df_man_study.reindex(sorted(df_man_study.columns), axis=1)
    df_aut_study= df_aut_study.reindex(sorted(df_aut_study.columns), axis=1)
    label= [a[0] for a in df_man_study.columns.str.split('_')]
    
    fig,ax = plt.subplots(2,2, sharex=True)
    fig.suptitle('Differences in exploration (aut-man)')
    for idx in range(0,df_man_test.columns.size):
        y_man_test= df_man_test.iloc[df_man_test.index.str.contains('Object3'), idx].values + df_man_test.iloc[df_man_test.index.str.contains('Object4'), idx].values
        y_aut_test= df_aut_test.iloc[df_aut_test.index.str.contains('Object3'), idx].values + df_aut_test.iloc[df_aut_test.index.str.contains('Object4'), idx].values
        ax[1,1].plot(range(1,7), y_aut_test-y_man_test, '-', label=label[idx])
        ax[1,1].set_title(' object 3+4 Test phase')
        
    
        y_man_test= df_man_test.iloc[df_man_test.index.str.contains('Object1'), idx].values + df_man_test.iloc[df_man_test.index.str.contains('Object2'), idx].values
        y_aut_test= df_aut_test.iloc[df_aut_test.index.str.contains('Object1'), idx].values + df_aut_test.iloc[df_aut_test.index.str.contains('Object2'), idx].values
        ax[1,0].plot(range(1,7), y_aut_test-y_man_test, '-', label=label[idx])
        ax[1,0].set_title(' object 1+2 Test phase')
        
        y_man_test= df_man_study.iloc[df_man_study.index.str.contains('Object3'), idx].values + df_man_study.iloc[df_man_study.index.str.contains('Object4'), idx].values
        y_aut_test= df_aut_study.iloc[df_aut_study.index.str.contains('Object3'), idx].values + df_aut_study.iloc[df_aut_study.index.str.contains('Object4'), idx].values
        ax[0,1].plot(range(1,7), y_aut_test-y_man_test, '-', label=label[idx])
        ax[0,1].set_title(' object 3+4 Study phase')
        
        y_man_test= df_man_study.iloc[df_man_study.index.str.contains('Object1'), idx].values + df_man_study.iloc[df_man_study.index.str.contains('Object2'), idx].values
        y_aut_test= df_aut_study.iloc[df_aut_study.index.str.contains('Object1'), idx].values + df_aut_study.iloc[df_aut_study.index.str.contains('Object2'), idx].values
        ax[0,0].plot(range(1,7), y_aut_test-y_man_test, '-', label=label[idx])
        ax[0,0].set_title(' object 1+2 Study phase')
        
        
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[1,1].legend(label, loc='center left', bbox_to_anchor=(1, 0.2))
    
    
    for a in ax.flat:
        a.set(label='minute',ylabel='Difference[s]', xlabel='minute')
        a.axhline(0, 0, color= 'k')
        a.axhline(-1,0, color='k')
        a.axhline(1,0, color='k')
        a.grid()
        
    plt.xticks([1,2,3,4,5,6])
    plt.tight_layout()
    print(df_man_study.columns, df_man_test.columns)
    plt.savefig(path_aut+'diff_obj_expl.png', bbox_inches= 'tight')
#%%  