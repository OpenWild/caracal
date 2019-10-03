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


### Library functions to match audio events from multiple loggers. This is used to determine
### time difference of arrivals from a single source to spatially distinct loggers.

### MultiLoggerEvent: Collection of individual Events, annotated with additional T(D)OA information
### - Reference logger - this is the logger that acts as the template for finding matches at the other
###                       loggers


#class MatchFn():
#    takes in a template stream and a candidate stream and calculates
#    the time and quality/confidence of the match, or will indicate if no accurate
#    match is found. This is intended as a generic function which can be subclassed
#    with specific implementations
#    returns matchDetails

import scipy
import numpy
import matplotlib.patches as patches
import pylab
from shumba import extract
from shumba import audiocore

## Generic class to estimate maximum delta time between two loggers, given the physical layout
# and knowledge of speed of sound etc. Here, we make a specific instantion, which just returns 
# a constant (fixed upper bound)
class FixedDelayBounder():

    def __init__(self,maxTime=5.0):
        self.maxTime=maxTime

    def estimateBound(self,loggerA,loggerB):
        return self.maxTime

## Generic class to estimate maximum delta time between two loggers, given the physical layout
# and knowledge of speed of sound etc. Here, we make a specific instantion, which just returns 
# a constant (fixed upper bound)
class PositionInformedDelayBounder():

    def __init__(self,propagationSpeed=344):
        self.propagationSpeed = propagationSpeed

    def estimateBound(self,loggerA,loggerB):
        posA = loggerA.position
        posB = loggerB.position
        distance = posA.distance(posB)
        maxTime = distance/self.propagationSpeed
        #print("distance",distance,"time",maxTime,loggerA.logger_id,loggerB.logger_id)
        return maxTime

## Dictionary like class/container that contains details of match event
class TDOArecord():
    """ dictionary like class that contains details for TDOA """

    def __init__(self,offsetTime,matchConfidence):
        self.offsetTime = offsetTime
        self.matchConfidence = matchConfidence

    def __repr__(self):
        mystr = "TDOA Record:\n"
        mystr += " Match Time Delay: " + str(self.offsetTime) + "\n"
        mystr += " Match Confidence: " + str(self.matchConfidence) + "\n"
        return mystr

class TimeDifferenceEstimator():
    
    def __init__(self,Fs=44100):
        self.Fs = Fs
        self.cached_segment = None
        self.Sxx = None
        
    def delta_time(self,segment,stream,segmentFs=44100,streamFs=44100,
                    low_band=40,high_band=600,fft_resolution=4096,fft_stride=384):
        """Returns delay in time samples. Can be negative.
        Low_band and high_band set a bandpass mask that can be used to focus
        on a signal of interest."""
        if self.cached_segment is not None:
            if (numpy.array_equal(self.cached_segment,segment)):
                Sxx = self.Sxx
            else:
                f,t,Sxx = scipy.signal.spectrogram(segment,fs=self.Fs,nperseg=fft_resolution,noverlap=fft_resolution-fft_stride,
                                mode='magnitude')
        else:
            f,t,Sxx = scipy.signal.spectrogram(segment,fs=self.Fs,nperseg=fft_resolution,noverlap=fft_resolution-fft_stride,
                                mode='magnitude')
        self.Sxx = Sxx
        self.cached_segment = segment

        f,t,Syy = scipy.signal.spectrogram(stream,fs=self.Fs,nperseg=fft_resolution,noverlap=fft_resolution-fft_stride,
                                mode='magnitude')
        # only deal with the frequency band of interest
        Sxx = Sxx[low_band:high_band,:]
        Syy = Syy[low_band:high_band,:]
        # detrend 
        Sxx = Sxx - numpy.mean(Sxx,axis=1,keepdims=True)
        Syy = Syy - numpy.mean(Syy,axis=1,keepdims=True)
        # normalize
        Sxx = Sxx/(numpy.sqrt(numpy.sum(Sxx**2)))
        Syy = Syy/(numpy.sqrt(numpy.sum(Syy**2)))
        # now do the overlap-add correlation between Sxx and Syy
        xc  = scipy.signal.correlate2d(Sxx, Syy,mode='valid')
        # flip around to do time-reversal matched filter (convolution)
        xc = numpy.fliplr(xc)
        delay = xc[0]
        # find the peak correlation
        pk_idx = numpy.argmax(delay)
        # midpoint
        mid = int(len(delay)/2)
        # convert to lag
        lag = pk_idx - mid + 1
        # convert lag to time in seconds
        lag = lag * fft_stride/self.Fs
        return lag,delay[pk_idx]

class SpectralMatcher():

    def __init__(self):
        self.td = TimeDifferenceEstimator()
        
    def matchStreams(self,refStream,candidateStream):
        delay, confidence = self.td.delta_time(refStream,candidateStream,fft_resolution=4096,
                        fft_stride=100)
        # create the TDOA record
        m = TDOArecord(delay,confidence)
        return m

class LoadAtTime():
    
    def __init__(self,Fs=44100):
        self.Fs=Fs
    
    def load(self,logger,fileTime,fineTime,length):
        audfile = audiocore.Beamformer(logger)
        audiodata,_,_ = audfile.beamformAtTime(fileTime)
        # now extract around the event
        fineSample = int(fineTime * self.Fs)
        lengthSample = int(length * self.Fs)
        # extract
        extract = audiodata[int(fineSample-lengthSample/2):int(fineSample+lengthSample/2)]
        return extract

class LoadAtTimeWithFs():
            
            def __init__(self,Fs=44100):
                self.Fs=Fs
            
            def load(self,logger,fileTime,fineTime,length):
                audfile = audiocore.Beamformer(logger)
                audiodata,_,_ = audfile.beamformAtTime(fileTime)
                realFs = len(audiodata)/100.0
                # now extract around the event
                fineSample = int(fineTime * self.Fs)
                lengthSample = int(length * self.Fs)
                # extract
                extract = audiodata[int(fineSample-lengthSample/2):int(fineSample+lengthSample/2)]
                return extract,realFs

## Dictionary like class/container that contains all the details of an event which
# has been matched across multiple loggers.
class MultiLoggerEvent():

    def __init__(self,referenceEvent):
        self.referenceEvent = referenceEvent
        self.matchedEvents = []

    def addEvent(self,matchEvent,matchDetails=None):
        self.matchedEvents.append([matchEvent,matchDetails])

    def __repr__(self):
        mystr = "MultiLogger Event Record:\n"
        mystr += ":::Reference Event:\n"
        mystr += str(self.referenceEvent)
        mystr += ":::Matched Events:\n"
        for item in self.matchedEvents:
            event = item[0]
            tdoa = item[1]
            mystr += str(event)   
            mystr += str(tdoa)        
        return mystr

    def plotMatches(self,loader,timespread=14.0):
        """This function plots the matched events across all the loggers, so the quality
        of matching can be visually inspected. Supply a loader function which then
        loads up the correct audio, given a logger index and timestamp. Optional
        timespread is the time around the central event"""
        # sort out our plotting context (bounds)
        fmin=0
        fmax=3000
        fsize=16 # label font size
        # first load up all the audio files and plot spectrograms
        pylab.figure(figsize=(16,12))
        numplots = len(self.matchedEvents)
        axislist = []
        plotindex = 1
        # This is the start time of the event
        centre_time = self.referenceEvent.event_time
        for event in self.matchedEvents:
            stream = loader(event[0].logger,event[0].coarse_timestamp,centre_time,timespread)
            ax = pylab.subplot(numplots,1,plotindex)
            axislist.append(ax)
            pylab.specgram(stream,Fs=44100,NFFT=4096,noverlap=3000)
            # relabel the x axis
            pylab.xticks(range(0,int(timespread),2), ["{:0.2f}".format(float(l)+centre_time-timespread/2) for l in range(0,int(timespread),2)])
            pylab.ylim(fmin,fmax)
            pylab.grid(True)
            # Add annotation to each panel
            pylab.text((timespread)*0.02,fmax*0.8,"Logger:"+str(event[0].logger.logger_id),fontsize=fsize,style='normal')
            # add box to each panel
            delta_time = event[0].event_time - centre_time + timespread/2 
            if event[0].logger.logger_id == self.referenceEvent.logger.logger_id:
                rect_color = 'red'
            else:
                rect_color = 'black'
            ax.add_patch(
                        patches.Rectangle(
                            ((delta_time,(fmax-fmin)*0.01)),
                            self.referenceEvent.event_length,
                            (fmax-fmin)*0.98,
                            edgecolor=rect_color,
                            fill=False      # remove background
                                 ) ) 
            str_time = "t={:.2f}s SS={:.2e}".format(event[0].event_time,event[0].SS)
            ax.text((timespread)*0.11,fmax*0.8,str_time, fontsize=fsize,style='normal')



            plotindex+=1
        if self.referenceEvent.classLabel is not None:
            pylab.suptitle(self.referenceEvent.classLabel)    
        pylab.tight_layout()


        
## Class which performs matching across multiple loggers
# It is supplied with concrete and user specific implementations of 
# - the matcher (class used to do the matching)
# - the loader (class used to load and optionally preprocess data, 
#      which could be raw, beamformed or in some other format)
# - the timebounder which sets the maximum time lag expected between any pair of loggers
class EventMatcher():

    def __init__(self,matcher,loader,timebounder):
        self.matcher = matcher # class that performs the matching
        self.loader = loader # method which loads data (could be raw or beamformed)
        # class which estimates the maximum time offset possible for 
        # a pair of loggers
        self.timebounder = timebounder 

    def matchEvents(self,referenceEvent,loggerList):
        """ This is a one-stop function to match as event across multiple loggers.
        Supply a referenceEvent, which will contain information about the 
        source logger and the time of the event.  """
        # create our multiLoggerEvent instance, based on the supplied event
        m = MultiLoggerEvent(referenceEvent)
        # load up our reference audio stream
        refLogger = referenceEvent.logger
        refCoarseTime = referenceEvent.coarse_timestamp
        refFineTime = referenceEvent.event_time
        refLength = referenceEvent.event_length
        referenceStream = self.loader(refLogger,refCoarseTime,refFineTime,refLength)
        # iterate over child loggers and compute match information
        for candidateLogger in loggerList:
            #if (candidateLogger.logger_id != refLogger.logger_id):
                # first we bound the searchLength according to the physical topology
                searchLength = self.timebounder.estimateBound(refLogger,candidateLogger)
                # we then load up a stream to match over
                candidateStream = self.loader(candidateLogger,
                                            refCoarseTime,
                                            refFineTime,
                                            refLength+(searchLength*2))
                matchDetails = self.matcher.matchStreams(referenceStream,candidateStream)
                if matchDetails is not None:
                    evCreator = extract.EventCreator()
                    # here we populate things like AoA and SS by creating a new event
                    matchEvent =  evCreator.createEvent(candidateLogger,
                                        refCoarseTime,
                                        refFineTime+matchDetails.offsetTime)
                    # and now we add it to the MultiLoggerEvent container
                    m.addEvent(matchEvent,matchDetails)    
        return m

        

    def matchEventsResampled(self,referenceEvent,loggerList):
        """ This is a one-stop function to match as event across multiple loggers.
        Supply a referenceEvent, which will contain information about the 
        source logger and the time of the event.  """
        # create our multiLoggerEvent instance, based on the supplied event
        m = MultiLoggerEvent(referenceEvent)
        # load up our reference audio stream
        refLogger = referenceEvent.logger
        refCoarseTime = referenceEvent.coarse_timestamp
        refFineTime = referenceEvent.event_time
        refLength = referenceEvent.event_length
        referenceStream,referenceFs = self.loader(refLogger,refCoarseTime,refFineTime,refLength)
        # iterate over child loggers and compute match information
        for candidateLogger in loggerList:
            #if (candidateLogger.logger_id != refLogger.logger_id):
                # first we bound the searchLength according to the physical topology
                searchLength = self.timebounder.estimateBound(refLogger,candidateLogger)
                # we then load up a stream to match over
                candidateStream,candidateFs = self.loader(candidateLogger,
                                            refCoarseTime,
                                            refFineTime,
                                            refLength+(searchLength*2))
                #print(referenceFs,candidateFs)
                matchDetails = self.matcher.matchStreams(referenceStream,referenceFs,
                                                        candidateStream,candidateFs)
                if matchDetails is not None:
                    evCreator = extract.EventCreator()
                    # here we populate things like AoA and SS by creating a new event
                    matchEvent =  evCreator.createEvent(candidateLogger,
                                        refCoarseTime,
                                        refFineTime+matchDetails.offsetTime)
                    # and now we add it to the MultiLoggerEvent container
                    m.addEvent(matchEvent,matchDetails)    
        return m