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


### Library functions to extract and enhance audio events from a single logger
# Event: Container for potential events
# EventSegmenter: multichannel based signal extraction using likelihood of AoA events

import numpy
from shumba import audiocore

import scipy.signal

# This is a container which holds all the information from a single logger about a (potential) event
class Event():
    
    def __init__(self,logger,coarse_timestamp,event_time,event_length=None):
        self.logger = logger # logger instance
        self.coarse_timestamp = coarse_timestamp # time in real-world time
        self.event_time = event_time # start time in real-world time (can be float)
        self.event_length = event_length # Optional: length of event
        self.AoA = None # Calculated later: Angle of Arrival
        self.SS = None # Calculated later: Signal Strength
        self.classLabel = None # Optional: if we know what sound it is

    def addAoA(self,corr,lag):
        self.AoA = [corr,lag]

    def addSS(self,signalstrength):
        self.SS = signalstrength

    def __repr__(self):
        mystr = "Event Record:\n"
        mystr += " Logger " + str(self.logger.logger_id) + ":" + str(self.logger.logger_name) +"\n"
        mystr += " File Time " + str(self.coarse_timestamp) +"\n"
        mystr += " Event Time " + str(self.event_time) +"\n"
        if self.event_length is not None:
            mystr += " Length: " + str(self.event_length) + "\n"
        if self.AoA is not None:
            mystr += " Delay Matrix (AoA):\n " + str(self.AoA[1]) + "\n"
        if self.SS is not None:
            mystr += " SS: {:.1e} ".format(self.SS) + "\n"
        if self.classLabel is not None:
            mystr += " Class: " + str(self.classLabel) + "\n"
        return mystr

class EventCreator():

    def __init__(self,window=4096,event_length=5.0,Fs=44100):
        self.window = window
        self.event_length = event_length
        self.Fs = Fs

    def createEvent(self,logger,coarse_timestamp,event_timestamp):
        """This function creates an arbitrary event at a particular timestamp and 
        populates the parameters """
        newEvent = Event(logger,coarse_timestamp,event_timestamp)
        # Now we have to load the correlation and lag values from file
        beamformer = audiocore.Beamformer(logger)
        beamformed, corr, lags = beamformer.beamformAtTime(coarse_timestamp,self.window)
        # now turn event_time into window offset
        # we need to add 1 to the windows to get a sensible AoA
        window_idx = int((event_timestamp*self.Fs)/self.window)
        sample_idx = int((event_timestamp*self.Fs))
        # add correlation details
        newEvent.addAoA(corr[window_idx+1],lags[window_idx+1])
        # calculate signal strength
        mid_len = int((self.event_length/2)*self.Fs)
        snippet = beamformed[sample_idx-mid_len:
                        sample_idx+mid_len]
        ss = audiocore.AudioStats.calcSS(snippet)
        newEvent.addSS(ss)
        return newEvent

class EventSegmenter():
    
    def __init__(self,window=4096,likelihood_threshold=-40,merge=25,max_lag=20,event_length=5.0,Fs=44100):
        self.window = window
        self.likelihood_threshold = likelihood_threshold
        self.merge = merge
        self.max_lag = max_lag
        self.event_length = event_length
        self.Fs = Fs
    
    def createMask(self,corr,lags):
        # turn lags into angles
        ac = audiocore.AudioCombine()
        angles,likelihoods = ac.computeAngles(corr,lags,self.max_lag)
        merge_counter = 0
        mask = []
        for a,l in zip(angles,likelihoods):
            if a is not None:
                if (l > self.likelihood_threshold):
                    if (merge_counter < 1):
                        mask.append(True)
                        merge_counter = self.merge
                    else:
                        mask.append(False)
                else:
                    mask.append(False)
            else:
                mask.append(False)
            merge_counter-=1
        return mask

    

    def maskToEvents(self,logger,coarse_timestamp,mask,beamformed,corr,lags):
        eventList = []
        sampleidx = 0
        for m,c,l in zip(mask,corr,lags):
            if m is True:
                e = Event(logger,coarse_timestamp,
                    event_time = float(sampleidx)/self.Fs,
                    event_length=self.event_length)
                e.addAoA(c,l)
                # calculate signal strength
                mid_len = int((self.event_length/2)*self.Fs)
                snippet = beamformed[sampleidx-mid_len:
                             sampleidx+mid_len]
                ss = audiocore.AudioStats.calcSS(snippet)
                e.addSS(ss)
                eventList.append(e)
            sampleidx+=self.window
        return eventList

    def extractEvents(self,logger,coarse_timestamp):
        """ This is a one-stop function to process a file and extract events """
        beamformer = audiocore.Beamformer(logger)
        beamformed, corr, lags = beamformer.beamformAtTime(coarse_timestamp,self.window)
        mask = self.createMask(corr,lags)
        # create events
        events = self.maskToEvents(logger,coarse_timestamp,mask,beamformed,corr,lags)
        return beamformed,events



class CoherentEventSegmenter():

    """This class uses the coherence metric (rather than beamforming) to extract 
    events"""
    
    def __init__(self,window=4096,coherence_window=512,
                coherence_threshold=5.0,merge=25,event_length=5.0,Fs=44100):
        self.window = window
        self.coherence_threshold = coherence_threshold
        self.merge = merge
        self.event_length = event_length
        self.Fs = Fs
        self.coherence_window = coherence_window

    def coherenceDetector(self,snippet_quad,window_size=512,fs=44100):
        """
        This function uses the coherence (rather than correlation) between channels
        as a metric of signal presence/absence.
        
        The window_size parameter here controls the PSD resolution"""
        f,Cxy = scipy.signal.coherence(snippet_quad[:,0],snippet_quad[:,1],fs=fs,nperseg=window_size)
        pk01 = numpy.max(Cxy)
        f,Cxy = scipy.signal.coherence(snippet_quad[:,0],snippet_quad[:,2],fs=fs,nperseg=window_size)
        pk02 = numpy.max(Cxy)
        f,Cxy = scipy.signal.coherence(snippet_quad[:,0],snippet_quad[:,3],fs=fs,nperseg=window_size)
        pk03 = numpy.max(Cxy)
        f,Cxy = scipy.signal.coherence(snippet_quad[:,1],snippet_quad[:,2],fs=fs,nperseg=window_size)
        pk12 = numpy.max(Cxy)
        f,Cxy = scipy.signal.coherence(snippet_quad[:,1],snippet_quad[:,3],fs=fs,nperseg=window_size)
        pk13 = numpy.max(Cxy)
        f,Cxy = scipy.signal.coherence(snippet_quad[:,2],snippet_quad[:,3],fs=fs,nperseg=window_size)
        pk23 = numpy.max(Cxy)
        metric = pk01+pk02+pk03+pk12+pk13+pk23
        return metric
    
    def createMask(self,audio):
        # turn window snippets into True/False labels
        # the merge parameter simply stops repeated events happening within 
        # a certain period, to stop repeated triggering
        merge_counter = 0
        mask = []
        for k in range(0,len(audio),self.window):
            snippet = audio[k:k+self.window,:]
            metric = self.coherenceDetector(snippet,
                            window_size = self.coherence_window,
                            fs = self.Fs)
            if (metric > self.coherence_threshold):
                if (merge_counter < 1):
                    mask.append(True)
                    merge_counter = self.merge
                else:
                    mask.append(False)
            else:
                mask.append(False)
            merge_counter-=1
        return mask

    def maskToEvents(self,logger,coarse_timestamp,mask,beamformed,corr,lags):
        eventList = []
        sampleidx = 0
        for m,c,l in zip(mask,corr,lags):
            if m is True:
                e = Event(logger,coarse_timestamp,
                    event_time = float(sampleidx)/self.Fs,
                    event_length=self.event_length)
                e.addAoA(c,l)
                # calculate signal strength
                mid_len = int((self.event_length/2)*self.Fs)
                snippet = beamformed[sampleidx-mid_len:
                             sampleidx+mid_len]
                ss = audiocore.AudioStats.calcSS(snippet)
                e.addSS(ss)
                eventList.append(e)
            sampleidx+=self.window
        return eventList

    def extractEvents(self,logger,coarse_timestamp,lpcutoff=2000.):
        """ This is a one-stop function to process a file and extract events """
        beamformer = audiocore.Beamformer(logger)
        beamformed, corr, lags = beamformer.beamformAtTime(coarse_timestamp,self.window)
        # load audio for the coherent detector
        opener = audiocore.AudioFile(logger)
        rawAudio = opener.loadAtTime(coarse_timestamp)
        util = audiocore.AudioUtility()
        filtAudio = util.detrend(rawAudio)
        hpAudio = util.highpassFilter(filtAudio)
        lpAudio = util.lowpassFilter(hpAudio,lpcutoff)
        # Extract the mask
        mask = self.createMask(lpAudio)
        # create events
        events = self.maskToEvents(logger,coarse_timestamp,mask,beamformed,corr,lags)
        return beamformed,events