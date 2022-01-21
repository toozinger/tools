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

from PyQt5 import QtWidgets, QtCore
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import sys  # We need sys so that we can pass argv to QApplication




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
 

# Main Program    
# ******************************************************************  ******************************************************************

# List what ports are available
serialPorts, serialPortDescriptions = checkComPorts()
print(f"Serial ports connected: {serialPorts}")
print(f"Serial port descriptions: {serialPortDescriptions}")
           
    



class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.graphWidget = pg.PlotWidget()
        self.setCentralWidget(self.graphWidget)

        self.x = [0, 1]
        self.y = [0,2]

        self.graphWidget.setBackground('w')

        pen = pg.mkPen(color=(255, 0, 0))
        self.data_line =  self.graphWidget.plot(self.x, self.y, pen=pen)
        
        # Timer for plotting
        self.plotTimer = QtCore.QTimer()
        self.plotTimer.setInterval(100)
        self.plotTimer.timeout.connect(self.update_plot_data)
        self.plotTimer.start()
        
        # Timer for reading serial
        self.serialReadTimer = QtCore.QTimer()
        self.serialReadTimer.setInterval(1)
        self.serialReadTimer.timeout.connect(self.readSerial)
        self.serialReadTimer.start()
        
        # Setup communication
        COMport = "COM4"
        baudRate = 115200
        
        self.saveFileLocal = "G:\My Drive\Documents\Projects\pythonToolsGithub"
        self.saveFileName = "TestSerial Read"
        
        txtSaveFileName = f"{self.saveFileName}.txt"
        self.saveFileFullName = os.path.join(self.saveFileLocal, txtSaveFileName)
        
        self.serialDataConnection = self.openSerial(COMport, baudRate)
        
        # Checkc if COM connected
        if not serialDataConnection: 
            sys.exit()
            
        # Initialization
        self.inWaiting = 0
        self.allData = []
     
    # Open serial port
    def openSerial(self, COMport, baudRate):
    
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


    def update_plot_data(self):
        
        self.data_line.setData(self.x, self.y)  # Update the data.

    
    # Function to read the serial data, if new data is available
    def readSerial(self):
        
        if self.inWaiting < self.serialDataConnection.in_waiting:
            try:
                data = self.serialDataConnection.readline().decode('unicode_escape').replace("\r\n","")            
            except:
                data = 0
                print("Error in reading serial line")
                
            data = self.cleanData(data)
            now = datetime.now()
            
            print(f"{now} {data}")
            self.allData.append([now, data])
            with open(self.saveFileFullName, "a") as saveFile: saveFile.write(f"{now}, {data}\n")
            
    def cleanData(self, data):
            
        dataList = data.split()
        cleanData = []
        
        for item in dataList:
            try:
                cleanData.append(float(item))
            except ValueError:
                pass
        
        if len(cleanData) == 1: cleanData = cleanData[0]
        
        return cleanData
    
    # Handeling close event nicer
    def closeEvent(self, event):
        
        self.serialDataConnection.close()
        
        # Optional save file to excel
        xlsSaveName = f"{self.saveFileLocal}\\{self.saveFileName}.xlsx"
        allDataDF = pd.DataFrame(self.allData)
        allDataDF.rename(columns={0: "dateTime", 1: "Data"})
        
        with pd.ExcelWriter(xlsSaveName) as writer:
            allDataDF.to_excel(writer, header=False, index=False)
         

app = QtWidgets.QApplication(sys.argv)
w = MainWindow()
w.show()
sys.exit(app.exec_())

