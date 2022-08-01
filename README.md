# EE18B036 Homework 1: Pulse Rate Estimation using Smartphone Camera footage.

There are 2 files for code
1. EE18B036_functions.py
2. EE18B036_HW1.ipynb

The first file is just a bag of functions
The second one involves the code used to generate plots, results and for tests.
For testing out the algorithm, please import 1.
For understanding the implementations, please go through 2., or the 'EE18B036_output_file.pdf' file attached.

This README file take you through all the functions used in homework 1 of CS6650, which one can access by importing the python file 'EE18B036_functions.py'

## Auxiliary functions

1. cleanSignal(x,w = 7)
Smoothens a signal of length l by moving average method. Return smoothened signal of lenght l-w+1

2. peakFreq(green, fps_act = 30, w=7)
Returns (number of peaks in the signal green)/(total time)

3. def dft_util(green, w=7)
returns the frequency and corresponding absolute DFT value of the signal 'green'

4. majorFreq(green,fps_act = 30,w1 = 7,w2=2, minPulse = 40, maxPulse = 200)
Returns frequency in BPM in the DFT of signal green of maximum magnitude, between minPulse and maxPulse.

5. dynamicOnly(frame1, frame2,cutoff = 5)
Returns selection (np.where type object) of frame where the absolute difference in frame1 and frame2 is greater than cutoff% of the maximum absolute difference


## IMPORTANT (as wanted in question 2)

1. BPM_dft(videoName,w1 = 7, w2 = 2)
Returns pulse rate in BPM from the full video using DFT technique. 
w1 and w2 are widths for cleaning signal and the DFT respectively.

2. BPM_ts(videoName,w = 7)
Returns pulse rate in BPM from the full video using Time series (TS) technique.
w is the window frame lenght to clean the signal.

3. plot_5sec(videoName)
Returns time and signal value for the video's first 5 seconds




## IMPORTANT (as wanted in question 3)
1. BPM_disp_ts(videoName,w = 7): 
Display video with pulse rate every second taken over 5 seconds using time series technique

2. BPM_disp_dft(videoName,w1 = 7,w2 = 2)
display video with pulse rate every second taken over 5 seconds using DFT technique

## IMPORTANT (as wanted in question 4)

1. BPM_sec_ts(videoName,t = 5,rescale = 100,fpsScale = 1, w = 7)
Return pulse rate for the first t seconds of video using time series technique
rescale is percentage of scaling frames and the fps is downsampled by fpsScale. 

2. BPM_sec_dft(videoName, t, rescale = 100,fpsScale = 1, w1 = 7,w2=2)
Return pulse rate for the first t seconds of video using DFT technique.
rescale is percentage of scaling frames and the fps is downsampled by fpsScale.

3. BPM_better_ts(videoName,t = 5, cutoff = 10,w0 = 7, w1 = 7)
Returns pulse rate for the first t seconds of video using time series technique and using dynamic pixels only
Cutoff is the percentage used in the auxiliary function dynamicOnly
w0 is for smoothening the signal before sent into peakFreq()

4. BPM_better_dft(videoName,t = 5, cutoff = 20,w0 = 10, w1 = 7, w2 = 2)
Returns pulse rate for the first t seconds of video using time series technique and using dynamic pixels only
Cutoff is the percentage used in the auxiliary function dynamicOnly
w0 is for smoothening the signal before sent into peakFreq()

5. plotClean(videoName, t = 5, w0 = 7, w1 = 7, w2 = 2, minPulse = 50, maxPulse = 200):
Plot the results for the performance of dynamic cropping.
Normal vs Dynamic cropping: Signal and DFT plots
