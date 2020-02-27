#!/usr/bin/env python
"""Image Analysis Pipeline
   Machine Learning Assisted Rice Hoja Blanca Disease Phenotyping using Multispectral UAV Imagery
   C. Delgado-Fajardo 
"""

from osgeo import gdal, gdal_array, ogr
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd
from sklearn.linear_model import LinearRegression
import cv2
from skimage.filters import threshold_otsu

#Create empty matrix
def createEmpty(y_sz, x_sz, n_band,datatype): 
    return np.zeros((y_sz, x_sz, n_band), gdal_array.GDALTypeCodeToNumericTypeCode(datatype))

#Read image and metadata
def readImg(imgDir):
    
    img_ds = gdal.Open(imgDir)
    x_sz = img_ds.RasterXSize
    y_sz = img_ds.RasterYSize
    n_band = img_ds.RasterCount
    ext = img_ds.GetGeoTransform()
    proj= img_ds.GetProjection()
    datatype = gdal.GDT_Float32

    
    img = createEmpty(y_sz, x_sz, n_band,datatype)

    for band in range(n_band):
        img[:, :, band] = img_ds.GetRasterBand(band + 1).ReadAsArray().astype("float32")/32768
        
    return img, x_sz, y_sz, n_band, ext, proj, datatype

#Shape to 
def rasterize(file_name, ysize, xsize, ext, proj, layer):
    
    src_ds = ogr.Open(file_name)
    src_lyr = src_ds.GetLayer()
    out_ds = gdal.GetDriverByName("MEM").Create('', xsize, ysize, 1, gdal.GDT_UInt32)
    out_ds.SetGeoTransform(ext)
    out_ds.SetProjection(proj)

    gdal.RasterizeLayer(out_ds, [1],
                        src_lyr,
                        None, None,  
                        [0], 
                        layer
                        )
    out_ds.FlushCache()

    return out_ds

# Normilized VI Function
def NVI(bandA, bandB):
    max_val = 1
    min_val = 0
    mask = np.greater(bandB + bandA, 0)
    VI = np.choose(mask, (0, (bandA - bandB) / (bandA + bandB)))
    VI[VI[:, :] > max_val] = max_val
    VI[VI[:, :] <= min_val] = np.nan
    return VI

# Radiometric Calibration
def ELC (img, roiELC, y_sz, x_sz, n_band, datatype):   
    # "X" - features
    # "y" - labels
    X = img[roiELC > 0, :] 
    y = roiELC[roiELC > 0]
    
    #Mean values across the panel
    ELC_dataset=np.column_stack((y,X))
    ELC_df = pd.DataFrame(ELC_dataset, columns=['Reference (%)', 'BLUE', 'GREN', 'RED', 'EDGE', 'NIR'])
    meanPanel=ELC_df.groupby('Reference (%)').mean().reset_index()
    meanPanel.loc[:, meanPanel.columns != 'Reference (%)']
    
    stdPanel=ELC_df.groupby('Reference (%)').std().reset_index()
    stdPanel.loc[:, meanPanel.columns != 'Reference (%)']
    
    #Regression theoretical (Panel) vs pixel base (Image)
    npMatrix = np.matrix(meanPanel)
    Y=npMatrix[:,0]
    m = []
    b = []
    for band in range(n_band):
        X=npMatrix[:,band+1]
        mdl = LinearRegression().fit(X,Y/100)
        m.append(mdl.coef_[0])
        b.append(mdl.intercept_)
    
    #ELC
    max_val = 1
    min_val = 0
    img_elc = np.zeros((y_sz, x_sz, n_band), gdal_array.GDALTypeCodeToNumericTypeCode(datatype))
    for band in range(n_band):
        img_elc[:, :, band]=((img[:, :, band])*m[band]+b[band]).astype(np.float32)
        dataOut= img_elc[:, :, band]
        # Enforce maximum and minimum values
        dataOut[dataOut[:, :] > max_val] = max_val
        dataOut[dataOut[:, :] < min_val] = min_val
        img_elc[:, :, band] = dataOut
    return img_elc

#Crop Masking
def mask (img_field):
    # RGB composite
    rgb_field = cv2.merge((img_field[:, :, 2], img_field[:, :, 1], img_field[:, :, 0]))
    
    # Gamma Correction
    gaussian_rgb = cv2.GaussianBlur(rgb_field, (9,9), 10.0)
    gaussian_rgb[gaussian_rgb<0] = 0
    gaussian_rgb[gaussian_rgb>1] = 1
    unsharp_rgb = cv2.addWeighted(rgb_field, 1.5, gaussian_rgb, -0.5, 0)
    unsharp_rgb[unsharp_rgb<0] = 0
    unsharp_rgb[unsharp_rgb>1] = 1
    
    gamma = 1.4
    gamma_corr_rgb = unsharp_rgb**(1.0/gamma)
    
    #Green Minus Red 
    GMR=gamma_corr_rgb[:, :, 1]-gamma_corr_rgb[:, :, 0]
    
    #Rice masking
    threshold_global_otsu = threshold_otsu(GMR)
    CC = GMR >= threshold_global_otsu
    
    return CC

#Local statistics
def Stats (roi, Indices, Names):   
    output = pd.DataFrame([])
    labels = np.unique(roi[roi > 0])
    output['ID']=labels
    i = 0
    for index, vi in enumerate(Indices):
        X = vi[roi > 0] 
        y = roi[roi > 0]
        dataset=np.column_stack((y,X))
        df = pd.DataFrame(dataset, columns=['ID', Names[i]])
        
        mean=df.groupby('ID').agg(np.nanmean).reset_index()
        std=df.groupby('ID').agg(np.nanstd).reset_index()
        
        output[Names[i]+'_MEAN']=mean[Names[i]]
        output[Names[i]+'_STD']=std[Names[i]]
        i = i + 1
        
    return output




