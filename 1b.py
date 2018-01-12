#This code imports a sound file (wav format), and takes a Fourier transform of the data into the frequency domain
#It then selectively removes the higher frequencies and outputs a new audio file with only the low frequencies remaining

#Special thanks to Paul Kushner for providing the basic code to make importing and exporting data easy to do 

#The scipy.io.wavfile allows you to read and write .wav files

from scipy.io.wavfile import read, write
from numpy import empty, linspace, arange, pi, where, zeros
import matplotlib.pyplot as plt
from numpy.fft import rfft, irfft
from pylab import subplot

#read the data into two stereo channels
#sample is the sampling rate, data is the data in each channel,
#dimensions [2, nsamples]

sample, data = read('C:\Users\Piotr\Documents\University\Computational Physics\Computational Assignment 5\GraviteaTime.wav')

#sample is the sampling frequency, 44100 Hz
#separate into channels

channel_0 = data[:,0]
channel_1 = data[:,1]

#Initialize the arrays, find max time using number of data points and sampling rate

N_points = len(channel_0)
t_max = N_points/44100.0
t = linspace(0,t_max,N_points)

#Plotting the raw data (time domain)

subplot(2,1,1)
plt.plot(t,channel_0)
plt.title('Channel 0')
plt.xlabel('Time (s)')
plt.ylabel('Data')

subplot(2,1,2)
plt.plot(t,channel_1)
plt.title('Channel 1')
plt.xlabel('Time (s)')
plt.ylabel('Data')

plt.show()

#Fourier Transforming, plotting initial transforms

fft_0 = rfft(channel_0)
fft_1 = rfft(channel_1)

freq = arange(N_points/2+1)*2*pi/t_max

subplot(2,1,1)
plt.title('Fourier Transform of Channel 0')
plt.plot(freq,abs(fft_0))
plt.xlabel('Frequency (Hz)')

subplot(2,1,2)
plt.title('Fourier Transform of Channel 1')
plt.plot(freq,abs(fft_1))
plt.xlabel('Frequency (Hz)')

plt.show()

#Filtering (Set frequencies greater than 880 Hz = 0)

Filter = 880

fft_0[where(freq >= Filter)] = zeros(len(where(freq >= Filter)))
fft_1[where(freq >= Filter)] = zeros(len(where(freq >= Filter)))

#Plot the new filtered FFT

subplot(2,1,1)
plt.title('Filtered Fourier Transform of Channel 0')
plt.plot(freq,abs(fft_0))
plt.xlabel('Frequency (Hz)')

subplot(2,1,2)
plt.title('Filtered Fourier Transform of Channel 1')
plt.plot(freq,abs(fft_1))
plt.xlabel('Frequency (Hz)')

plt.show()

#Inverse Transforming, plotting new filtered signal (time domain)

Filtered_channel_0 = irfft(fft_0)
Filtered_channel_1 = irfft(fft_1)

subplot(2,1,1)
plt.plot(t,Filtered_channel_0)
plt.title('Filtered Channel 0')
plt.xlabel('Time (s)')
plt.ylabel('Data')

subplot(2,1,2)
plt.plot(t,Filtered_channel_1)
plt.title('Filtered Channel 1')
plt.xlabel('Time (s)')
plt.ylabel('Data')

plt.show()

channel_0_out = Filtered_channel_0
channel_1_out = Filtered_channel_1

#this creates an empty array data_out with the
#same shape as "data" (2 x N_Points) and the same
#type as "data" (int16)

data_out = empty(data.shape, dtype = data.dtype)

#fill data_out
data_out[:,0] = channel_0_out
data_out[:,1] = channel_1_out
write('GraviteaTime_lpf.wav', sample, data_out)

