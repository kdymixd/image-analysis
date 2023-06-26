# -*- coding: utf-8 -*-
"""
Created on Tue May 24 

@author: Maxime LECOMTE

Code to analyze datas
"""

import datetime
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os, fnmatch
import re
import pandas as pd
from analysis_abstract import AnalysisNoFit, AnalysisPropagation, AnalysisSigma, AnalysisSigma2
from open_picture import get_absorption_picture, select_roi
import pyarrow as pa
import pyarrow.parquet as pq 
import platform

"""
Make sure to have the right backend on VScode or Spyder to run this program. On sypder, go to preferences
and choose "Qt5" in "graphics"
"""

plt.ioff()

"""Parameters"""

_date=datetime.date(2023,5,24)
save_file_name="A0.txt"
run_name= "A0"
frame_name="frames_0000"
use_custom_float=False
effective_pixel_size=1.57e-6 
analyzer=AnalysisSigma(effective_pixel_size, bin=1)
thermal=True
C_sat=np.inf

do_ROI_selection=True
do_background_selection=True
cam_name="Pixelfly"
# ROI=np.s_[361:716,500:882]
ROI=None
background_roi=None
angle=None

"""Functions"""
# Because Cicero uses ',' as a separator for floats and not '.'

def custom_float(s):
	return(float(s.replace(',','.')))

def dico_date(date, dico):
    """Dictionnary function associating a month to the right Cicero folder name
    Dico = False -> month folder name
    Dico = True -> day folder name"""
    
    year=date.strftime("%Y")
    month=date.strftime("%m")
    day=date.strftime("%d")
    dict_list=[["01","Jan"], ["02","Feb"], ["03","Mar"], ["04","Apr"], ["05","May"], ["06","Jun"], ["07","Jul"], ["08","Aug"], ["09","Sep"], ["10","Oct"], ["11","Nov"], ["12","Dec"]]
    
    for e in dict_list:
        if month in e:
            if dico==False:
                return e[1]+year
            else:
                return day + e[1] + year
            break

def find_log(file_path, runlogs_path):
    """
    Associate an image to its log file
    """
    
    cdate=time.ctime(os.path.getctime(file_path))
    cdate=time.strptime(cdate)
    pattern_cdate=str(time.strftime("%H%M%S", cdate))

    #We take into account the potential delay between the creation time of 2 files
    ds=2 #Tolerance in the creation time in seconds
    pattern_cdate_extended=[]
    for i in range(ds+1):
        pattern_cdate_extended.append(str(int(pattern_cdate)+i))
        pattern_cdate_extended.append(str(int(pattern_cdate)-i)) 
    
    #Searching the corresponding log file
    result = []
    for root, dirs, files in os.walk(runlogs_path):
        for name in files:
            log_ctime=time.ctime(os.path.getctime(os.path.join(root, name)))
            log_ctime=time.strptime(log_ctime)
            pattern_clog=str(time.strftime("%H%M%S", log_ctime))
            if pattern_clog in pattern_cdate_extended:
                result.append(os.path.join(root, name))
                break

    return result
    
def extract_cicero_data(log_path):
    """
    Extract the variables scanned and their values
    """
    
    file_name_ext=os.path.basename(log_path) # Filename with its extension
    a=os.path.splitext(file_name_ext)
    
    # Remove the .clg extension
    file_name=""
    for i in range(len(a)-1):
        file_name+=a[i]
    
    # Separate variables and values
    variable_name=file_name.split('_')
    splitted_name=[]
    for e in variable_name:
        splitted_name.append(e.split(" = "))
    return splitted_name

"""
Get paths
"""

passerelle_path="Z:/"
path_raw_data = passerelle_path + "Data/" + _date.strftime("%Y") + "/" + _date.strftime("%m") + "/" + _date.strftime("%d") + "/" + run_name + "/"
# path_raw_data = os.path.join(path_raw_data, run_name)
path_raw_data_frame = path_raw_data + frame_name + '.tiff'
# path_raw_data_frame = os.path.join(path_raw_data, frame_name)
path_analyzed_data = passerelle_path + "Data_Analysis/" + _date.strftime("%Y") + "/" + _date.strftime("%m") + "/" + _date.strftime("%d") + "/" + "Py_analysis/" + run_name + "/"
# path_analyzed_data = os.join(path_analyzed_data, run_name)
# path_analyzed_data = os.join(path_analyzed_data, frame_name)
path_runlogs = passerelle_path + "Cicero/" + _date.strftime("%Y") + "/" + dico_date(_date, False) + "/" + dico_date(_date, True) + "/" + "RunLogs/"
first_time = True

"""
Regex to find the Cicero variables and their values
"""

# float_separator = '\.'
# pattern_for_variables=re.compile("(?:#([^#=]+)\s=\s([\d" + float_separator+"\+\-]+))")
# pattern_for_timestamp=re.compile("(^\d{6})")

"""
Check the existence of files, to avoid another analysis
"""

data_path = os.path.join(path_analyzed_data, frame_name)
data_exists=os.path.isfile(data_path)

roi_file_exists=os.path.isfile(os.path.join(path_analyzed_data, "roi.txt"))

#If it does we open the dataframe
if data_exists:
	table=pq.read_table(data_path)
	df=table.to_pandas()
    
if roi_file_exists:
	with open(os.path.join(path_analyzed_data, "roi.txt"),"r") as file:
		roi_string=file.readline().split(",")
		ROI=np.s_[int(roi_string[0]):int(roi_string[1]), int(roi_string[2]): int(roi_string[3])]
		do_ROI_selection=False
        
#We scan the folder
list_dir=list(os.scandir(path_raw_data))

try :
    
    # #Create the .txt file for the image analysis
    # try:
    #     path_analyzed_frame_txt=path_analyzed_data + frame_name + '.txt'
    #     try:
    #         os.mkdir(path_analyzed_data)
    #         print('Frame folder created')
    #     except Exception:
    #         print('Frame folder already created')
        
    #     #Create columns in the txt file
    #     if not os.path.exists(path_analyzed_frame_txt):
    #         with open(path_analyzed_frame_txt, 'w') as analyzed_file:
    #             columns=['Frame name','Log name']
    #             for e in columns:
    #                 analyzed_file.write(e)
    #                 analyzed_file.write('\t')
    #         print('Txt file created')
    #     else:
    #         print('Txt file already created')
            
    # except Exception as e:
    #     print(e)
    #     print('\t Problem creating the .txt file')
    
    
    for frames in list_dir:
        
        #Find the log and variables associated to the frame
        frame_log=find_log(frames.path, path_runlogs)
        
        #We open the image
        if first_time:
            if do_ROI_selection or do_background_selection:
                if do_ROI_selection:
                    # ROI=select_roi(os.path.join(path, folder.name), camera_name=cam_name, name="Select ROI")
                    ROI=select_roi(frames.path, path_analyzed_data, frame_name, camera_name=cam_name, name="Select ROI")
                if do_background_selection:
                    # background_roi=select_roi(os.path.join(path, folder.name), camera_name=cam_name, name="Select background")
                    background_roi=select_roi(frames.path, path_analyzed_data, frame_name, camera_name=cam_name, name="Select background")
                else:
                    background_roi=None
                analyzer.create_plot()
            else:
                analyzer.create_plot()

        # img=get_absorption_picture(os.path.join(path, folder.name), camera_name=cam_name, roi_background=background_roi, C_sat=C_sat)
        img=get_absorption_picture(frames.path, path_analyzed_data, frame_name, camera_name=cam_name, roi_background=background_roi, C_sat=C_sat)

        if ROI is not None:
            img=img[ROI]
        else:
            img=img[50:]
                
		#We create a dictionnary to store the parameters 
        log_path=find_log(frames.path, path_runlogs)
        var_name=extract_cicero_data(log_path[0])
        
        metadata={}
        metadata['Sequence']=var_name[1][0]
        
        for i in range(2,len(var_name)):
            metadata[var_name[i][0]]=float(var_name[i][1])
                
		#We analyze data
        timestamp='Test'
        analyzed_dic=analyzer.plot_and_save(img, timestamp, frame_name, thermal=thermal)
        parameters_dic={**metadata, **analyzed_dic} #We merge metadata with data from analysis
        print("Parameters_dic : ", parameters_dic)
        
        if not data_exists:
            df=pd.DataFrame([parameters_dic], index=[timestamp])
            data_exists=True
        else:
            df.loc[timestamp]=parameters_dic
        if first_time:
            first_time=False
    
                
except KeyboardInterrupt or Exception as e:
    print(e)
    pq.write_table(pa.Table.from_pandas(df), os.path.join(path_analyzed_data, save_file_name))

# =============================================================================
# pq.write_table(pa.Table.from_pandas(df), os.path.join(path, save_file_name))
# #save ROI
# if ROI is not None: 
# 	with open(os.path.join(path, "roi.txt"), "w") as file: 
# 		file.write(f"{ROI[0].start}, {ROI[0].stop}, {ROI[1].start}, {ROI[1].stop}\n")
# =============================================================================

print(df.columns)

 #Generate dataset using Dataframe object from pandas. Then save it using parquet_pyarrow which seems to be the fastest format.