# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 17:34:51 2021

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
from datetime import datetime




# Read in file
root = tk.Tk()
root.withdraw()
ctypes.windll.user32.MessageBoxW(0, "Select the file for analysis", "Selection 1", 1)
dataFilePath = filedialog.askopenfilename(title = "Select the file")
if not dataFilePath: sys.exit()

fileName = dataFilePath.split("/")[-1].split(".")[0]
saveLocal = "/".join(dataFilePath.split("/")[0:-1])


fileType = dataFilePath.split("/")[-1].split(".")[-1]
colNames = ["X", "Y"]

# Parse if Excel file
if "xlsx" in fileType:
    df = pd.read_excel(dataFilePath, header=None)
    dataDF = df.rename(columns={0: "X", 1: "Y"})
    
startTime = dataDF["X"][0]

# Convert timestamp column to running time
dataDF["X"] = dataDF["X"].apply(lambda x: (x-startTime).total_seconds())

# # Parse if Text file
# elif "txt" in fileType:
#     with open(dataFilePath, encoding="UTF-8") as fp:
#         rawData = fp.readlines()
    
#     data = []
#     for line in rawData:
#         line = line.strip("\n").split()[-1] # Get second item, that is comma separated
        
#         line = [float(x) for x in line.split()] # Extract numbers only
        
#         data.append(line)
        
    
#     dataDF = pd.DataFrame(data, columns = colNames)

# else:
#     print("Unsure of file format??")
#     sys.exit()


# Start performing auto-slope measuring anallysis
y = dataDF["Y"].to_numpy()
yhat = savgol_filter(y, 151, 3) # data smoothing

x = dataDF["X"].to_numpy()

# First derivitive  and smoothing
dy = np.gradient(yhat,0.5)
dyhat = savgol_filter(dy, 5, 3)

# Second derivitive and smoothing
# ddy = np.gradient(dyhat, 0.5)
# ddyhat = savgol_filter(ddy, 251, 3)

# Pick end points of linear region, with optional modifiers
endModify = 1         # Essentially minimum amount of data you want for linear region
startModify = 1       # Make sure you're away from beginning oddities
dyMaxModifier = 1     # Denotes the % of height max of 2nd derivitie, as used to find starting index 

# Find start index
percentile = 0.1 # Indicates how much the first derivitive should spike before calling this the start
startIndex = np.where(dyhat > max(dyhat)*percentile)[0][0]

# Flips array around, and does the same tacx for finding where the derivitive starts to change
dyhatReversed = dyhat[::-1]
endIndex = len(x) - np.where(dyhatReversed > max(dyhatReversed)*percentile)[0][0]

# Function for iterating through to optimize r^2
def optimizeEndPoint(startIndex, endIndex):
    
    # Create best fit line within current region of intereest        
    m, b = np.polyfit(x[startIndex:endIndex], y[startIndex:endIndex], 1)
    
    # Calculate load curve and line curve coordinate arrays
    lineCoords = x[startIndex:endIndex]*m + b
    curveCoords = y[startIndex:endIndex] 
    
    # Perform coorelation with r^2 analysis
    correlation_matrix = np.corrcoef(lineCoords, curveCoords)
    correlation_xy = correlation_matrix[0,1]
    r_squared = correlation_xy**2
    
    return r_squared, startIndex, endIndex

r_squared, startIndex, endIndex = optimizeEndPoint(startIndex, endIndex)

bringBackIterations = 0
pushForwardIterations = 0
prev_objective = -1
objective = r_squared

 # Objective looping function to optimized by bringing in the end point
while objective > prev_objective:
    prev_objective = objective
    endIndex -= 1
    r_squared, startIndex, endIndex = optimizeEndPoint(startIndex, endIndex)
    objective = r_squared
    bringBackIterations += 1
    
 # Objective looping function to optimized by pushing out the start point
prev_objective = -1
while objective > prev_objective:
    prev_objective = objective
    startIndex += 1
    r_squared, startIndex, endIndex = optimizeEndPoint(startIndex, endIndex)
    objective = r_squared
    pushForwardIterations += 1

m, b = np.polyfit(x[startIndex:endIndex], y[startIndex:endIndex], 1)
r_squared, startIndex, endIndex = optimizeEndPoint(startIndex, endIndex)



# Use calculated indices for slope calculation
yStart =    dataDF["Y"].iloc[startIndex]
xStart =    dataDF["X"].iloc[startIndex]

yEnd =      dataDF["Y"].iloc[endIndex]
xEnd =      dataDF["X"].iloc[endIndex]

# Slope of tangent
slope = (yEnd - yStart)/(xEnd - xStart)


print(f"Filename: {fileName}")
print(f"pushForwardIterations: {pushForwardIterations}")
print(f"bringBackIterations: {bringBackIterations}")
print(f"R^2 value: {r_squared}")

print(f"Delta Y: {yEnd - yStart}")
print(f"Delta X: {xEnd - xStart}")
print(f"Slope delY/delX: {slope}")

print(f"Slope via deltaY/30: {(yEnd - yStart)/30}")

index = 1
fig1, ax1 = plt.subplots()
fig1.canvas.manager.set_window_title(f"File: {fileName}")
# ax1.plot(x, y)
ax1.plot(x, yhat)
# ax1.plot([x[startIndex], x[endIndex]], [y[startIndex], y[endIndex]], "b-*")
# ax1.plot(x[startIndex:endIndex], lineCoords)
ax1.plot(x[startIndex:endIndex],x[startIndex:endIndex]*m + b)
ax1.set_xlabel("X")
ax1.set_ylabel("Y")


# fig2, ax2 = plt.subplots()
# ax2.plot(x, dyhat)
# ax2.plot(x, ddyhat)
# ax2.plot(x, dddyhat)



