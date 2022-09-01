# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 10:44:43 2022

@author: Lenovo
"""
import matplotlib.pyplot as plt

#4*4 plot

def plot(data = 0):
    fig, axs = plt.subplots(4,4)
    rows, cols = 4,4
    
    for row in range(rows):
        for col in range(cols):
            axs[row,col].plot(data)
    
plot(data = [2,3,6,4,5])