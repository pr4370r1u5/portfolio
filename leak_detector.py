
##############################################
###     LEAK DETECTOR ALGO                 ###
###     CS6750 Spring 2023                 ###
###     Mobile & Ubiquitous Computing      ###
###                                        ###
###     TEAM WATERBEARS                    ###
###     written by: Richard Praetorius     ###
###     Md Azmain Amin                     ###
###     Erik Bakke                         ###
###     Sushmitha Berdere                  ###
##############################################

import numpy as np
import scipy as sp
import itertools as it
#import math
#import copy
#import pyodbc # communicates with SQL

import csv

# possible time parsing libraries
#import time
#import calendar
#from datetime import timedelta

# visualize man
#import matplotlib.pyplot as plt


### ASSUMPTIONS: ###

# at least 2 days of water meter data is available, maximum of 4 months
# resolution of analysis is TEN MINUTES, change in time_keeper __init__
# data is analyzed day-by-day to maximize consistency

### END ASSUMPTIONS ###


class leak_detector:
    
    def __init__(self, time_stamps):

        self.time_stamps = time_stamps
        
        #minutes
        self.minute = 1
        self.moment = 3
        self.jiffy = 5
        self.two_shakes = 10
        self.a_bit = 30
        self.hour = 60
        self.eighth = 180
        self.day = 1440
        self.week = 10080
        self.month = 43200

        self.time_after_time = [self.minute, self.moment, self.jiffy, self.two_shakes, self.a_bit, self.hour, self.eighth, self.day, self.week, self.month]
        # self.time_after_time = [1,10,100,1000,10000,100000]
        # time after time...
        # time after time...

        #daysina
        self.months = [31,28,31,30,31,30,31,31,30,31,30,31]

        
        ###  CHANGE RESOLUTION HERE  ###
        
        self.resolution = self.two_shakes

        ###  END CHANGE RESOLUTION   ###


    def epoch_convert(self, time_stamp):
        #INPUT:  time_stamp string 2023-03-23T14:23:05xxx
        #RETURN: time in seconds since 2020-01-01T00:00:00xxx

        '''
        daylight savings
        second sunday in March - fwd
        first sunday in November - back
        
        DONE: Check INT size
        
        31,104,000 seconds in a year
        518,400 minutes in a year

        32 bit max val 2,147,483,647 = 70 years
        16 bit max val 65,535

        '''

        yr = int(time_stamp[:4])
        mn = int(time_stamp[5:7])
        dy = int(time_stamp[8:10])
        h = int(time_stamp[11:13])
        m = int(time_stamp[14:16])
        s = int(time_stamp[17:19])

        segundos = 0

        # years
        segundos += (yr-2020) * 365 * self.day * 60

        while yr >= 2020:
            if yr%4 == 0:
                dy+=1
                yr-=4
            else:
                yr-=1

        # months
        k = 1
        while k<mn:
            segundos += self.months[k] * self.day * 60
            k+=1
        
        # days + hms
        segundos += dy*self.day*60 + h*self.hour*60 + m*60 + s

        return segundos
    

    def alert(self, event = 0): #main function
        #input:  NOTHING (list of strings - time_stamps)
        #output: binary - ALERT status based on data

        # max intermittent flow of water meter = 50GPM

        # check resolution in ASSUMPTIONS above
        # possible arrays, depending on resolution:
        #    minute, moment, jiffy, two_shakes, a_bit, hour, day, week, month


        ## HISTOGRAM ##
        self.time_stamps.sort()
        time_list = [self.epoch_convert(self.time_stamps[0]), 1] #starts histogram at position [1]
        time_interval = self.resolution*60
        n = 1

        for time_stamp in self.time_stamps:
            now_what = self.epoch_convert(time_stamp)
            now_what -= time_list[0]

            while now_what >= time_interval*n:
                time_list.append(0)
                n+=1
            
            time_list[n]+=1

        time_list = time_list[1:]



        ###   ADD SIMULATED LEAK DATA   ###

        if event == 1:

            time_list = self.small_drip_generator(time_list) #adds steady drip to data

        elif event == 2:
        
            time_list = self.big_leak_generator(time_list) #adds big gush to last half hour of data

        elif event == 3:
        
            time_list = self.bath_time_generator(time_list) #adds 25 gallon bath instance


        ### END ADD SIMULATED LEAK DATA ###

        
        
        ## MOVING AVERAGE ##
        # https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy
        
        #find and stack moving averages
        
        fat_stacks = np.array(time_list).reshape((1,n,1))

        for minit in self.time_after_time:
            if n > minit and minit > self.resolution:
                q = np.convolve(fat_stacks[0,:,0], np.ones(minit), 'valid') / minit 
                q = np.concatenate((np.zeros(minit-1),q))        
                
                fat_stacks = np.concatenate((fat_stacks, q.reshape((1,n,1))), axis = 2)
                
                        
        
        enc = np.shape(fat_stacks) # one row, histogram bin columns, moving averages depth
        day_length = int(self.day/self.resolution) #whole number of one day of bins
        
        whole_days = int(n // day_length)
        remaining_days = int(n % day_length)
        
        #centers current data by adding zeros for half the day
        
        np.concatenate( (fat_stacks, np.zeros((enc[0], int(n//(day_length*2)), enc[2])) ), axis = 1 )

        #reshapes into cube: row = day, column = bin, depth = moving average
        
        
        
        chunky = fat_stacks[:,remaining_days:,:].reshape((whole_days, day_length, enc[2])) 

        #take averages
        training_data = np.sum(chunky[:whole_days-1, :, :], axis = 0) / (whole_days-1) #average the columns
        test_data = chunky[-1,:,:] #current day's data
   


        ## COMPARE DATA ##

        #just checks if data is greater than the moving average data
        #then adds together the % overshot

        #the nature of a leak is not if it starts, but IF IT STOPS

        threshold = 0.0

        checker = list(test_data[:,0])
        hyucker = list(training_data[:,0])

        # moving averages crossing
        for c,h in zip( checker, hyucker ):
            
            testit = (c-h)/(h+0.0001)
            
            if testit < 0 or c == 0:
                threshold = 0
            else:
                threshold += testit*0.01
        

        # fft
        # find the small drip
        # https://numpy.org/doc/stable/reference/generated/numpy.fft.fft.html

        freaq_out = np.fft.fft(time_list)

        peaky = sp.signal.find_peaks(freaq_out.real, height = 200, prominence = 150 )

        udder = np.fft.fftfreq(len(time_list))

        for n in range(np.shape(peaky[0])[0] - 3):
            threshold += 0.2


        if threshold >= 1:
            return True
        else:
            return False



    def small_drip_generator(self, time_list):
           
        # adds steady drip - one gallon every five hours
        

        five_hours = 300/self.resolution
        
        for n in range(len(time_list)):
            if n % five_hours == 0:
                time_list[n] += 1
        
        return time_list
    

    def big_leak_generator(self, time_list):

        # big leak - greater than 2 gallons/minute sustained
        # set quantity below

        big_leak = 2 * self.resolution

        day_length = int(self.day/self.resolution)
        
        interval = day_length//48
        #interval = day_length//24
        #interval = day_length//3

        for n in range(-1, -1*interval, -1):
            time_list[n] += big_leak 
        
        return time_list
    

    def bath_time_generator(self, time_list):

        # 5 minutes at 5 gallons/minute = 25 gallons

        time_list[-1] += 25
        
        return time_list



def main(time_stamps):
    
    alert = False
    ticks = leak_detector(time_stamps)
    alert = ticks.alert() # run the primary function
    
    return alert

if __name__ == "__main__":
    # if this is the main program, then run this
    # otherwise it is imported elsewhere and not run
    
    # https://learn.microsoft.com/en-us/sql/connect/python/pyodbc/step-3-proof-of-concept-connecting-to-sql-using-pyodbc?view=sql-server-ver16

    # https://docs.python.org/3/library/csv.html
    # https://realpython.com/python-csv/

    ## TEST CODE ##

    time_stamps = []

    with open('water_meter_dataUTC.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            time_stamps.append(row[0])

    print(main(time_stamps))