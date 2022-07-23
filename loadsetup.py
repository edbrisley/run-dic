import os
from configparser import ConfigParser
import numpy as np
import cv2 as cv
##load the configuration file containing the DIC settings
def LoadSettings(image_set):
    configur = ConfigParser()
    configur.read('Setup.ini')
    sub_size = configur.getint('Subsets','SubsetSize')
    sub_frequency = configur.getint('Subsets','SubsetFrequency')
    GF_stddev = configur.getfloat('Filters','GaussianFilterStdDev')
    GF_filtsize = configur.getint('Filters','GaussianFilterSize')
    SF_order = configur.getint('Miscellaneous','ShapeFunctionOrder')
    corr_refstrat = configur.getint('Miscellaneous','CorrelationReferenceStrategy')
    calibrate = configur.getint('Miscellaneous','Calibration')
    #store setup parameters
    settings = np.array([])
    settings = np.append(settings,sub_size)
    settings = np.append(settings,SF_order)
    settings = np.append(settings,GF_stddev)
    settings = np.append(settings,GF_filtsize)
    settings = np.append(settings,corr_refstrat)
    settings = np.append(settings,sub_frequency)
    #define images working directory
    img_0 = image_set[0]
    img_0 = cv.imread(img_0,0)
    img_rows = img_0.shape[0]
    img_columns = img_0.shape[1]
    settings = np.append(settings,img_rows)
    settings = np.append(settings,img_columns)
    settings = np.append(settings,corr_refstrat)
    settings = np.append(settings,calibrate)
    return settings
##load the images from directory
def LoadImages():
    current_working_directory = os.getcwd()
    folder = '\images'
    image_location = current_working_directory + folder
    image_set = []
    for filename in os.listdir(image_location):    
        image_set.append(filename)
    return image_set
##printout details of settings and images
def SettingsInfo():
#return a summary of the correlation and image setup
    pass