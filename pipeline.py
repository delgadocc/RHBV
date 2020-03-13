#!/usr/bin/env python
"""Image Analysis Pipeline
   Machine Learning Assisted Rice Hoja Blanca Disease Phenotyping using Multispectral UAV Imagery
   C. Delgado-Fajardo 
"""

# Import libraries
from osgeo import gdal, gdal_array, ogr
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import math
import pandas as pd
import cv2
from skimage.filters import threshold_otsu
from sklearn.linear_model import LinearRegression
from sklearn.metrics import recall_score
from IPython.display import display_html
from scipy.stats import pearsonr

#Create empty matrix
def createEmpty(y_sz, x_sz, n_band,datatype): 
    return np.zeros((y_sz, x_sz, n_band), gdal_array.GDALTypeCodeToNumericTypeCode(datatype))

#Read image and metadata
def readImg(imgDir):
    #Read image and metadata
    img_ds = gdal.Open(imgDir)
    x_sz = img_ds.RasterXSize
    y_sz = img_ds.RasterYSize
    n_band = img_ds.RasterCount
    ext = img_ds.GetGeoTransform()
    proj= img_ds.GetProjection()
    datatype = gdal.GDT_Float32
    #Allocate memory
    img = createEmpty(y_sz, x_sz, n_band,datatype)
    #Create the matrix
    for band in range(n_band):
        img[:, :, band] = img_ds.GetRasterBand(band + 1).ReadAsArray().astype("float32")/32768
        
    return img, x_sz, y_sz, n_band, ext, proj, datatype

#Shape to raster
def rasterize(file_name, ysize, xsize, ext, proj, layer):
    #Read the files
    src_ds = ogr.Open(file_name)
    src_lyr = src_ds.GetLayer()
    out_ds = gdal.GetDriverByName("MEM").Create('', xsize, ysize, 1, gdal.GDT_UInt32)
    out_ds.SetGeoTransform(ext)
    out_ds.SetProjection(proj)
    #Rasterize
    gdal.RasterizeLayer(out_ds, [1],
                        src_lyr,
                        None, None,  
                        [0], 
                        layer
                        )
    out_ds.FlushCache()

    return out_ds

#Normalized Vegetation Indices
def NVI(bandA, bandB, min_val, max_val):
    # Avoid division by zero
    mask = np.greater(bandB + bandA, 0)
    VI = np.choose(mask, (0, (bandA - bandB) / (bandA + bandB)))
    # Enforce maximum and minimum values
    VI[VI[:, :] > max_val] = np.nan
    VI[VI[:, :] <= min_val] = np.nan
    
    return VI

# Soil-based VIs
def soilVIs(R_soil, NIR_soil, R_crop, NIR_crop, soil): 
    #WDVI    
    mask = np.greater(R_soil, 0)
    a = np.choose(mask, (0, (NIR_soil) / (R_soil)))
    WDVI = NIR_crop - a * R_crop
    #PVI
    R_filt = R_soil[soil > 0]
    NIR_filt = NIR_soil[soil > 0]
    mdl = LinearRegression().fit(R_filt.reshape(-1, 1),NIR_filt.reshape(-1, 1))
    a = mdl.coef_[0]
    b = mdl.intercept_
    PVI = a*NIR_crop - R_crop/math.sqrt(math.pow(a,2)+1)
    # MSAVI
    MSAVI2 = (2 * (NIR_crop + 1) - np.sqrt((2 * NIR_crop + 1)**2 - 8 * (NIR_crop - R_crop)))/2
    
    return WDVI, PVI, MSAVI2

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
    #Empirical Line Calibration
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
    #Thresholding
    threshold_global_otsu = threshold_otsu(GMR)
    CC = GMR > threshold_global_otsu
    soil = GMR <= threshold_global_otsu
    return CC, soil

#Local statistics
def Stats (roi, Indices, Names):
    #Establish the labels
    output = pd.DataFrame([])
    labels = np.unique(roi[roi > 0])
    output['ID']=labels
    #i = 0
    #Iterate over the indices
    for index, vi in enumerate(Indices):
        X = vi[roi > 0] 
        y = roi[roi > 0]
        dataset=np.column_stack((y,X))
        df = pd.DataFrame(dataset, columns=['ID', Names[index]])
        df = df.replace(0, np.nan)
        #Stats
        mean=df.groupby('ID').agg(np.nanmean).reset_index()
        std=df.groupby('ID').agg(np.nanstd).reset_index()
        #Results
        output[Names[index]+'_MEAN']=mean[Names[index]]
        output[Names[index]+'_STD']=std[Names[index]]
        #i = i + 1
        
    return output

#Penalized function
def fpreproc(dtrain, dtest, param):
    label = dtrain.get_label()
    ratio = float(np.sum(label == 0)) / np.sum(label == 1)
    param['scale_pos_weight'] = ratio
    return (dtrain, dtest, param)

#ML framework
def ML(data, model, train=None):
    # Data structure to store the metrics
    results = {'DATASET':list(), 'AUC':list(), 'TN':list(), 'TP':list(), 'AVG':list()}
    DATASET, AUC = list(), list()
    TN, TP, AVG = list(), list(), list()
    data_grouped = data.groupby('DATASET')
    #Iterate over datasets
    for name, group in data_grouped:
        # If train
        if train is not None: 
            #train set
            train = group[group.TYPE=='variety']
            train = train.drop(['TYPE','DATASET'], axis=1)
            X_train = train.drop('CLASS', axis=1)
            y_train = train.CLASS
            #fit the model
            model = model.fit(X_train,y_train)
        else:
            pass
        #validation set
        test = group[group.TYPE=='line']
        test = test.drop(['TYPE','DATASET'], axis=1)
        X_test = test.drop('CLASS', axis=1)
        y_test = test.CLASS
        #predict
        predicted = model.predict(X_test)
        #metrics
        DATASET.append(name)
        AUC.append(model.best_score_)
        TN.append(recall_score(y_test, np.round(predicted), average=None)[0])
        TP.append(recall_score(y_test, np.round(predicted), average=None)[1])
        AVG.append(recall_score(y_test, np.round(predicted), average='weighted'))
    #store metrics    
    results['DATASET'] = list(DATASET)
    results['AUC'] = list(AUC)
    results['TN'] = list(TN)
    results['TP'] = list(TP)
    results['AVG'] = list(AVG)
    results = pd.DataFrame.from_dict(results)   
    #stats
    mean = results[['AUC','TN','TP','AVG']].mean().values
    std = results[['AUC','TN','TP','AVG']].std().values
    #store the stats
    results = results.append({'DATASET' : 'mean' , 'AUC' : mean[0], 'TN' : mean[1], 'TP' : mean[2], 'AVG' : mean[3]} , ignore_index=True)
    results = results.append({'DATASET' : 'std' , 'AUC' : std[0], 'TN' : std[1], 'TP' : std[2], 'AVG' : std[3]} , ignore_index=True)
    #display results
    display_html(results.round(2).to_html(index=False), raw=True)
    
    return results.round(2)

#P-values
def calculate_pvalues(df):
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            pvalues[r][c] = round(pearsonr(df[r], df[c])[1], 4)
    return pvalues

def calculate_corr(df):
    correl = df.corr()
    correl = correl.round(2)        
    pval = calculate_pvalues(df)
    
    #Create the masks
    r1 = correl.applymap(lambda x: '{}*'.format(x))
    r2 = correl.applymap(lambda x: '{}**'.format(x))
    r3 = correl.applymap(lambda x: '{}***'.format(x))
    
    #Apply the masks
    correl = correl.mask(pval< 0.1,r1)
    correl = correl.mask(pval< 0.05,r2)
    correl = correl.mask(pval< 0.01,r3)

    return correl