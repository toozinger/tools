# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 11:34:01 2021

@author: dowdt
"""
import tkinter as tk
from tkinter import filedialog
import os

import pandas as pd
import ctypes
import sys
import matplotlib.pyplot as plt 
from math import pi
from scipy.signal import savgol_filter
import numpy as np

# Read in first test data file
root = tk.Tk()
root.withdraw()
ctypes.windll.user32.MessageBoxW(0, "Select the first datafile in a folder of datafiles", "Selection 1", 1)
dataFilePath = filedialog.askdirectory(title = "Select the location of the first datafile")
if not dataFilePath: sys.exit()

folderDir = "/".join(dataFilePath.split("/")[0:-1])

dataFiles = []

for file in sorted(os.listdir(folderDir)):
    if ".dat" in file: dataFiles.append(file)


# Read in constants file
root = tk.Tk()
root.withdraw()
ctypes.windll.user32.MessageBoxW(0, "Select the file containing all of the constants", "Selection 2", 1)
constantsFilePath = filedialog.askopenfilename(title = "Select the constants file")
if not constantsFilePath: sys.exit()

constants = pd.read_excel(constantsFilePath)


# function to read data file and parse it into a pandas dataframe
def readDatafile(fileLocal):
    
    with open(fileLocal) as fp:
        rawData = fp.readlines()

    # Read in data, skipping non-number rows
    rawData = rawData[5:-1]
    
    testData = []
    for row in rawData:
        if len(row.split()) == 3:
            try:
                dataRow = [float(x) for x in row.split()]
                testData.append(dataRow)
            except:
                pass
    colNames = ["N", "mm", "s"]
    dataDF = pd.DataFrame(testData, columns = colNames)
    
    return dataDF

# *****************************************************************************
# Function for doing analysis
def flexural(fileLocal, fileConstants, index):

    dataDF = readDatafile(fileLocal)
    
    
    # Constants
    span = fileConstants["Span"]            # [mm]
    width = fileConstants["Width"]          # [mm]
    depth = fileConstants["Depth"]          # [mm]
    diameter = fileConstants["Diameter"]    # [mm]
    testName = fileConstants["Name"]
    
    # Invert column sign convention
    dataDF["N"] = dataDF["N"].apply(lambda x: x*-1)
    dataDF["mm"] = dataDF["mm"].apply(lambda x: x*-1)
    
    
    # Find Maxes
    maxForce = dataDF["N"].max()
    maxForceIndex = dataDF["N"].idxmax()
    dispAtMaxForce = dataDF["mm"].loc[maxForceIndex]
    
    # Start performing auto-slope measuring anallysis
    y = dataDF["N"].to_numpy()
    yhat = savgol_filter(y, 151, 3) # data smoothing
    
    x = dataDF["mm"].to_numpy()
    
    # First derivitive  and smoothing
    dy = np.gradient(yhat,0.5)
    dyhat = savgol_filter(dy, 551, 3)
    
    # Second derivitive and smoothing
    # ddy = np.gradient(dyhat, 0.5)
    # ddyhat = savgol_filter(ddy, 251, 3)
    
    # Pick end points of linear region, with optional modifiers
    endModify = 0.7         # Essentially minimum amount of data you want for linear region
    startModify = 1.1       # Make sure you're away from beginning oddities
    dyMaxModifier = 0.5     # Denotes the % of height max of 2nd derivitie, as used to find starting index 
    
    # Find start index
    startIndex = int(np.where(dyhat > max(dyhat)*dyMaxModifier)[0][0]*startModify)
    
    # Slice second 3/4 of data for finding ending, to remove weird start derivitive dips
    dyhatMinusStart = dyhat[int(len(dyhat)/4):-1]
    if min(dyhatMinusStart) < 0:
        endIndex = int(((len(dyhat)/4) + np.where(dyhatMinusStart < 0)[0][0])*endModify)
    else:
        endIndex = int(len(dyhat)*endModify)
    
    # Function for iterating through to optimize r^2
    def optimizeEndPoint(endIndex):
        
        # Create best fit line within current region of intereest        
        m, b = np.polyfit(x[startIndex:endIndex], y[startIndex:endIndex], 1)
        
        # Calculate load curve and line curve coordinate arrays
        lineCoords = x[startIndex:endIndex]*m + b
        curveCoords = y[startIndex:endIndex] 
        
        # Perform coorelation with r^2 analysis
        correlation_matrix = np.corrcoef(lineCoords, curveCoords)
        correlation_xy = correlation_matrix[0,1]
        r_squared = correlation_xy**2
        
        return r_squared, endIndex
    
    r_squared, endIndex = optimizeEndPoint(endIndex + 1)  
    
    iterations = 0
    prev_objective = -1
    objective = r_squared
    
    # Objective looping function. Keeps increasing range of load curve used for line fit until it no longer improves the r^2 value
    while objective > prev_objective:
        prev_objective = objective
        endIndex += 1
        r_squared, endIndex = optimizeEndPoint(endIndex)
        objective = r_squared
        iterations += 1
    
    # Since last loop will reduce the r^2, go back one iteration for the max
    endIndex -+ 1
    m, b = np.polyfit(x[startIndex:endIndex], y[startIndex:endIndex], 1)
    r_squared, endIndex = optimizeEndPoint(endIndex)
    
    print(iterations)
    print(r_squared)
    
    # Plot various aspects after optimization, and see the best fit line that was picked
    fig1, ax1 = plt.subplots()
    fig1.canvas.manager.set_window_title(f"Test {testName}")
    # ax1.plot(x, y)
    ax1.plot(x, yhat, "g-", label = "Test Data")
    ax1.plot(x[startIndex:endIndex], x[startIndex:endIndex]*m + b, "b-", label = "Curve Fit")
    ax1.title.set_text(f"""Best fit slope line for test: {testName} """)
    ax1.legend()
    ax1.set_xlabel("Displacement [mm]")
    ax1.set_ylabel("Force [N]")
    
    # Use calculated indices for slope calculation
    dispStart =     dataDF["mm"].iloc[startIndex]
    forceStart =    dataDF["N"].iloc[startIndex]
    
    dispEnd =       dataDF["mm"].iloc[endIndex]
    forceEnd =      dataDF["N"].iloc[endIndex]

    # Slope of tangent [N/mm] 
    slope = (forceEnd - forceStart)/(dispEnd - dispStart)
    
    # strength [MPa]
    # Auto choose circular or rectangular cross section, depending on which is filled out in constants spreadsheet
    if pd.isna(diameter):
        strength = 3*maxForce*span/(2*width*depth**2)
    else:
        strength = maxForce*span**3/(4*diameter**4)
        
    # Stiffness [GPa]
    if pd.isna(diameter):
        stiffness = span**3*slope/(4*width*depth**3)/1000
    else:
        stiffness = maxForce*span**3/(48*dispAtMaxForce*pi*diameter**4/64)

    return strength, stiffness, dataDF
# *****************************************************************************

   
# ***** Call function
strengths = []
stiffnesses = []
dataDFs = []

# Where to start and end the linear slope analysis
startDisp = 1   # [mm]
endDisp = 4     # [mm]

for index, file in enumerate(dataFiles):
    
    fileLocal = folderDir + "/" + file
    fileConstants = constants.iloc[index]
    
    strength, stiffness, dataDF = flexural(fileLocal, fileConstants, index)
    
    strengths.append(strength)
    stiffnesses.append(stiffness)
    dataDFs.append(dataDF)
    
# Save data
constants["Strength"]=strengths
constants["Stiffness"]=stiffnesses

saveLocal = "/".join(constantsFilePath.split("/")[0:-1])
saveName = constantsFilePath.split("/")[-1].split(".")[0] + "_output.xlsx"


constants.to_excel(f"{saveLocal}/{saveName}",  index=False,)

# Plot load displacement
fig1, ax1 = plt.subplots()
fig1.canvas.manager.set_window_title("Combined Test Data")
for i, file in enumerate(dataFiles):
    ax1.plot(dataDFs[i]["mm"], dataDFs[i]["N"], label = f"""Test {constants["Name"].iloc[i]} """)
ax1.legend()
ax1.set_xlabel("Displacement [mm]")
ax1.set_ylabel("Force [N]")


# Plot strengths
fig2, ax2 = plt.subplots()
fig2.canvas.manager.set_window_title("Test Strengths")
ax2.plot(np.linspace(1,len(constants),len(constants)), strengths, "*")
ax2.title.set_text("Strengths")
ax2.set_ylim([0, max(strengths)*1.1])
ax2.set_xlabel("Test")
ax2.set_ylabel("Flexural Strength [MPa]")
ax2.set_xticks(np.linspace(1,len(constants),len(constants)), constants["Name"])
ax2.tick_params(labelrotation=45)

# Plot stiffnesses
fig3, ax3 = plt.subplots()
fig3.canvas.manager.set_window_title("Test Stiffnesses")
ax3.plot(np.linspace(1,len(constants),len(constants)), stiffnesses, "*")
ax3.title.set_text("Stiffnesses")
ax3.set_ylim([0, max(stiffnesses)*1.1])
ax3.set_xlabel("Test Number")
ax3.set_ylabel("Flexural Stiffness [GPa]")
ax3.set_xticks(np.linspace(1,len(constants),len(constants)), constants["Name"])
ax3.tick_params(labelrotation=45)
