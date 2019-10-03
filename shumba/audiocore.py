#
#                                _ _     _ 
#                               (_) |   | |
#  ___  _ __   ___ _ ____      ___| | __| |
# / _ \| '_ \ / _ \ '_ \ \ /\ / / | |/ _` |
#| (_) | |_) |  __/ | | \ V  V /| | | (_| |
# \___/| .__/ \___|_| |_|\_/\_/ |_|_|\__,_|
#      | |                                 
#      |_|                                 
#
# part of the openwild project

### Core audio functions including:

# AudioFile: file handling - reading and writing wavefiles
# AudioUtility: filtering and detrending
# AudioStats: correlation based statistics
# AudioCombine: Multichannel beamforming/angle of arrival extraction
# AudioViz: Spectral visualization/plotting of audio data

import numpy
import pylab
import os
from scipy.io import wavfile
from scipy.fftpack import rfft, irfft, fftfreq, fft, ifft
import scipy.signal
from scipy.signal import butter, lfilter
from scipy.signal import freqs

from shumba import physicalmodels
from shumba import shumbacore

class AudioFile():
    
    def __init__(self,logger,sampling_rate=44100):
        self.logger = logger
        self.sampling_rate = sampling_rate

    def _load_file(self,full_filename):
        _,data = wavfile.read(full_filename)
        return data

    def loadAtTime(self,timestamp):
        sub_dir = self.logger.filetemplate.format(self.logger.logger_id,timestamp)
        filename = os.path.join(self.logger.directory,sub_dir)
        data = self._load_file(filename)
        return data

    def save_file(self,data,filename,sub_dir=None):
        if sub_dir is None:
            full_filename = os.path.join(self.logger.directory,filename)
        else:
            # check that the subdir exists, else create it
            full_dir = os.path.join(self.logger.directory,sub_dir)
            if not (os.path.isdir(full_dir)):
                os.makedirs(full_dir)
            full_filename = os.path.join(full_dir,filename)
        data = numpy.asarray(data, dtype=numpy.float)
        wavfile.write(full_filename,self.sampling_rate,data)

class AudioUtility():
    
    def __init__(self,sampling_rate=44100):
        self.sampling_rate = sampling_rate
    
    def highpassFilter(self,data, filter_stop_freq=10,filter_pass_freq=50,filter_order=801):
        # High-pass filter (was 40,80)
        nyquist_rate = self.sampling_rate / 2.
        desired = (0, 0, 1, 1)
        bands = (0, filter_stop_freq, filter_pass_freq, nyquist_rate)
        filter_coefs = scipy.signal.firls(filter_order, bands, desired, nyq=nyquist_rate)
        # Apply high-pass filter
        filtered_audio = scipy.signal.filtfilt(filter_coefs, [1], data,axis=0)
        return filtered_audio
        
    def butter_lowpass(self,cutOff, fs, order=5):
            nyq = 0.5 * fs
            normalCutoff = cutOff / nyq
            b, a = butter(order, normalCutoff, btype='low', analog = False)
            return b, a

    def lowpassFilter(self,data, cutOff, order=4):
        fs = self.sampling_rate 
        b, a = self.butter_lowpass(cutOff, fs, order=order)
        y = scipy.signal.filtfilt(b, a, numpy.array(data),axis=0)
        return y
    
    def detrend(self,data):
        return data - numpy.mean(data)
    
    def normalize(self,data):
        data = self.detrend(data)
        scale_factor = numpy.max(numpy.abs(data))*2.0
        data = data/scale_factor
        return data,scale_factor

class AudioStats():
    
    @staticmethod
    def calc_xcorr(s1,s2):
        """Calculate the cross-correlation between two signals"""
        xc  = scipy.signal.fftconvolve(s1, s2[::-1], mode='full')
        return xc

    @staticmethod
    def calc_gcc_xcorr(s1,s2):
        """ Calculate the generalization cross-correlation between two signals
        - This is called the 'phase accurate' transform, as it ignores amplitude information"""
        pad1 = numpy.zeros(len(s1))
        pad2 = numpy.zeros(len(s2))
        s1 = numpy.hstack([s1,pad1])
        s2 = numpy.hstack([pad2,s2])
        f_s1 = fft(s1)
        f_s2 = fft(s2)
        f_s2c = numpy.conj(f_s2)
        f_s = f_s1 * f_s2c
        denom = numpy.abs(f_s)
        # sort out div by zero issues
        denom[denom < 1e-100] = 1e-100
        f_s = f_s / denom  # This line is the only difference between GCC-PHAT and normal cross correlation
        return numpy.abs(ifft(f_s))[1:]

    @staticmethod
    def calcSS(s1):
        """This calculates the power/signal strength of a supplied snippet"""
        if len(s1) > 0:
            rms = numpy.sqrt(numpy.mean(s1**2))
        else:
            rms = 0.0
        return rms


class AudioCombine():
    
    def __init__(self,num_channels=4):
        self.num_channels = num_channels
        # fake world
        world = shumbacore.World([])
        # common aoa model
        self.aoaModel = physicalmodels.AoA_Model(world)
        
    def computeLagMatrix(self,data,mode='normal'):
        """Calculate the tdoa matrix over a segment of data, returning strength and values

        - The mode flag sets whether we use GCC_PHAT or normal cross correlation
        @TODO we need to update this function to take advantage of the symmetry
        of the lag matrix.
        This will provide a significant speed boost.

        """
        corr_pk = []
        corr_mx = []
        for idx_i in range(self.num_channels):
            corr_pk_row = []
            corr_row = []
            for idx_j in range(self.num_channels):
                a = numpy.squeeze(data[:,idx_i])
                b = numpy.squeeze(data[:,idx_j])
                if mode == 'GCC':
                    xc = AudioStats.calc_gcc_xcorr(a,b)
                else:
                    xc = AudioStats.calc_xcorr(a,b)
                xc_pk = numpy.argmax(xc)
                xc_max = numpy.max(xc)
                #print(xc_pk,xc_max)
                corr_row.append(xc_pk)
                corr_pk_row.append(xc_max)
            corr_mx.append(corr_row)
            corr_pk.append(corr_pk_row)
        midpoint = numpy.size(data,axis=0)
        delay_mx = (midpoint - numpy.array(corr_mx)-1)
        if mode == 'GCC':
            delay_mx = delay_mx
        return numpy.array(corr_pk),delay_mx
    

    def beamformSegment(self,segment,pk,mx):
        """Input a segment (N * k) audio
        a delay matrix (k * k)
        and the correlation weights (k * k)
        to compute delay-and-sum beamforming"""
        mixed_wave = segment[:,0]
        for k in range(1,numpy.shape(segment)[1]):
            mixed_wave = mixed_wave + numpy.roll(segment[:,k],mx[0][k])
        return mixed_wave

    def computeAngles(self,corr,lags,max_delay=20):

        angles,likelihoods = self.aoaModel.estimate_angles(lags,max_delay)
        return angles,likelihoods


    def beamform(self,data,window=4096,output='beamform'):
        """ Outputs:
        beamform: just the beamformed audio
        raw: beamformed audio, lags and correlations matrices"""
        beamformed = numpy.zeros(numpy.shape(data)[0])
        num_windows = int(numpy.shape(data)[0]/window)
        correlations = []
        lags = []
        for k in range(num_windows):
            # compute the lags for a segment
            start_beamform_idx = int(k*window)
            stop_beamform_idx = int((k+1)*window)
            segment = data[start_beamform_idx:stop_beamform_idx,:]
            corr, lag = self.computeLagMatrix(segment)
            # we now use a slightly wider segment, filtered with a triangular window
            # this overlaps the beamformed segments so they fade neatly into one another
            start_wide = int(numpy.max([start_beamform_idx-window*2,0]))
            stop_wide = int(numpy.min([stop_beamform_idx+window*2,numpy.shape(data)[0]]))
            wide_segment = data[start_wide:stop_wide,:]
            beamformed_segment = self.beamformSegment(wide_segment,corr,lag)
            filtered_segment = scipy.signal.triang(len(beamformed_segment))*beamformed_segment
            beamformed[start_wide:stop_wide] = beamformed[start_wide:stop_wide] + filtered_segment
            # save
            correlations.append(corr)
            lags.append(lag)
        if output == 'raw':
            return beamformed,correlations,lags
        if output == 'beamform':
            return beamformed
        raise("Error: not a correct output type")
        return None


class AudioViz():
    
    def __init__(self,Fs=44100):
        self.Fs = Fs
        
    def plotSpectrogram(self,data,show=False):
        # Plot 4 channel spectrogram
        if len(numpy.shape(data)) > 1:
            if numpy.shape(data)[1] == 4:
                for k in range(numpy.shape(data)[1]):
                    pylab.subplot(numpy.shape(data)[1],1,k+1)
                    pylab.specgram(data[:,k],Fs=self.Fs,NFFT=4096,noverlap=3000)
                    pylab.ylim(0,2000)
        else:
            pylab.specgram(data,Fs=self.Fs,NFFT=4096,noverlap=3000)
            pylab.ylim(0,2000)
        if show is True:
            pylab.show()

    def plotSpectrolines(self,
                          data, 
                          spacing=0.05,
                          low_band=10,
                          high_band=200,
                          step=2):
        f,t,Sxx = scipy.signal.spectrogram(data,fs=self.Fs,nperseg=16384,noverlap=16000,
                                mode='magnitude')
        # work out the max range
        Sxx_max = numpy.max(Sxx[low_band:high_band,:])
        Sxx_max = Sxx_max * step
        # plot
        for band in range(low_band,high_band,step):
            spec = Sxx[band:band+step,:]
            spec_total = numpy.sum(spec,axis=0)/Sxx_max
            norm = spec_total - numpy.mean(spec_total)
            pylab.plot(norm+(band-low_band)*spacing,'r',alpha=0.8)
        pylab.xlim(0,len(t))
        pylab.ylim([0,spacing*(high_band-low_band)])

class Beamformer():

    def __init__(self,logger):
        self.logger = logger

    def beamformAtTime(self,timestamp,window=4096):
        # sort out the paths to load up what we need
        lagfile = str(self.logger.logger_id)+"_" +str(timestamp)+".bf"
        lagdir = os.path.join(self.logger.directory,"beamformed")
        full_lag_filename = os.path.join(lagdir,lagfile)
        wavfile = str(self.logger.logger_id)+"_" +str(timestamp)+".wav"
        wavdir = os.path.join(self.logger.directory,"beamformed")
        full_wav_filename = os.path.join(wavdir,wavfile)
        # we check if the files already exist
        if os.path.exists(full_lag_filename) and os.path.exists(full_wav_filename):
            pickfile = shumbacore.Pickler(self.logger.directory)
            frame = pickfile.load(str(self.logger.logger_id)+"_" +str(timestamp)+".bf","beamformed")
            corr = frame[0]
            lags = frame[1]
            # load the audio itself
            audfile = AudioFile(self.logger)
            bf = audfile._load_file(full_wav_filename)
            # all done
            return bf,corr,lags
        # load file
        audfile = AudioFile(self.logger)
        audiodata = audfile.loadAtTime(timestamp)
        # preprocess file
        util = AudioUtility()
        avedata = util.detrend(audiodata)
        hpdata = util.highpassFilter(avedata)
        # beamform file
        ac = AudioCombine()
        beamformed, corr, lags = ac.beamform(hpdata,window,output='raw')
        # save the beamformed file
        audfile.save_file(beamformed,str(self.logger.logger_id)+"_" +str(timestamp)+".wav","beamformed")
        # save the meta-data
        pickfile = shumbacore.Pickler(self.logger.directory)
        pickfile.save([corr,lags],str(self.logger.logger_id)+"_" +str(timestamp)+".bf","beamformed")
        # return the data to the caller
        return beamformed,corr,lags

