# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 13:02:42 2022

@author: dowdt
"""

from datetime import datetime
import time
import serial
import serial.tools.list_ports
import sys
import pandas as pd
import os

# Function for checking what COM ports are connected, and which are open
# ******************************************************************
def checkComPorts():
    rawPorts = serial.tools.list_ports.comports()   
    
    serialPorts = []
    serialPortDescriptions = []
    
    # find the open ports, and their hwids
    for portNumber, portDescription, VID_PID in sorted(rawPorts):
          serialPorts.append(portNumber)
          serialPortDescriptions.append(portDescription) 
          
    return serialPorts, serialPortDescriptions

# Function for opening up the arduino serial port
# ******************************************************************
def openSerial(COMport, baudRate):
    
    # Try to open connection
    try:
        serialDataConnection = serial.Serial(COMport,baudRate,timeout=0.1, write_timeout=0.5)
        serialDataConnection.flushInput()
        time.sleep(0.2) # wait for Arduino
        # print("Controller Arduino Connected")
        return serialDataConnection
    
    # Attemps to self-fix issue from un-closed ports
    except:
        pass
        # print("Failed to connect to Arduino. Trying again")
        
    # try to close the serial if it's open
    try:
        serialDataConnection.close()   
    except:
        pass
       
    try:
        serialDataConnection = serial.Serial(COMport,baudRate,timeout=5)
        serialDataConnection.flushInput()
        time.sleep(2) # wait for Arduino
        # print("Controller Arduino Connected")
        return serialDataConnection
    finally:
        print("Failed to connect. Serial busy with another program")
        return False

        
# Small function for preparing and sending data to the arduino
# ******************************************************************
def sendData(serialDataConnection, controlString, controlFloat):
    
    try:
        # Applies line delimeters, and puts the comma between the words and values
        sendData = "<"+ str(controlString) + "," + str(controlFloat) + ">" 
        #print("sending: ", sendData)
        serialDataConnection.write(sendData.encode('utf-8'))
        exitNow = False
        return exitNow
    except KeyboardInterrupt:
        print("Manual exit requested")
        exitNow = True
        return exitNow
    except:
        print("Error in sending serial data to Arduino")
 
# Function for reading the arduino data       
# ******************************************************************
def readController(serialDataConnection):
    
        
    try:
        serialLine = serialDataConnection.readline().decode('unicode_escape').replace("\r\n","")
    
    except KeyboardInterrupt:
        serialDataConnection.close()
        
    except:
        serialLine = 0
        print("Error in reading arduino line")
        
    return serialLine

# Main Program    
# ******************************************************************  ******************************************************************

# List what ports are available
serialPorts, serialPortDescriptions = checkComPorts()
print(f"Serial ports connected: {serialPorts}")
print(f"Serial port descriptions: {serialPortDescriptions}")

# Manually select COM port
COMport = "COM4"
baudRate = 115200

saveFileLocal = "G:\My Drive\Documents\Projects\pythonToolsGithub"
saveFileName = "TestSerial Read"

txtSaveFileName = f"{saveFileName}.txt"
saveFileFullName = os.path.join(saveFileLocal, saveFileName)

serialDataConnection = openSerial(COMport, baudRate)

if not serialDataConnection: 
    sys.exit()

# Checks buffer for new data
inWaiting = 0

allData = []

# Infinite loop for reading data
while True:
    
    # Checks if new data, and if so, reads it.
    try:
        if inWaiting < serialDataConnection.in_waiting:
            data = readController(serialDataConnection)
            now = datetime.now()
            
            print(f"{now} {data}")
            allData.append([now, data])
            with open(saveFileFullName, "a") as saveFile: saveFile.write(f"{now}, {data}\n")
    
    except KeyboardInterrupt:
        serialDataConnection.close()
           
    
# Optional save file to excel
xlsSaveName = f"{saveFileLocal}\\{saveFileName}.xlsx"
allDataDF = pd.DataFrame(allData)
allDataDF.rename(columns={0: "dateTime", 1: "Data"})

with pd.ExcelWriter(xlsSaveName) as writer:
    allDataDF.to_excel(writer, header=False, index=False)
