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

import numpy
import pylab
import math
from copy import deepcopy

class DensePositionEstimator():

    def __init__(self,world,timeDev=0.1,numPoints=1000,matchThreshold=0.05):
        self.world = world
        self.timeDev = timeDev
        self.numPoints = numPoints
        self.matchThreshold = matchThreshold
        self.maxTime = 9.0 # don't look for solutions beyond this, penalize very heavily

    @staticmethod
    def likelihood(mu, sigma, x):
        """calculates the probability of x for 1-dim Gaussian with mean mu and var. sigma
        @param mu:    predicted distance
        @param sigma: standard deviation
        @param x:     observed distance
        @return gaussian value
        """
        # calculates the probability of x for 1-dim Gaussian with mean mu and var. sigma
        return numpy.exp(- ((mu - x) ** 2) / (sigma ** 2) / 2.0) / numpy.sqrt(2.0 * numpy.pi * (sigma ** 2))

    def estimateTimeOfEmission(self,matchRecord,testX,testY):
        """ Given a test location, calculate the emission time for each of the
        loggers. Broadly, if the location is correct, then all the emission times
        should be identical."""
        estTimes = []
        for match in matchRecord.matchedEvents:
            logX,logY = match[0].logger.position.xy()
            dist = math.hypot(logX-testX,logY-testY)
            time = dist/self.world.SOUND_SPEED
            estEventTime = match[0].event_time - time
            estTimes.append(estEventTime)
        estTimes = numpy.array(estTimes)
        return estTimes

    def estimateTimeOfArrival(self,matchRecord,testX,testY):
        estTimes = self.estimateTimeOfEmission(matchRecord,testX,testY)
        # Option 1: Use median time of emission as reference
        midTime = numpy.median(estTimes)
        # Option 2: Use the reference event as a fixed fiducial
        refLoggerID = matchRecord.referenceEvent.logger.logger_id
        for match,eTime in zip(matchRecord.matchedEvents,estTimes):
            if refLoggerID == match[0].logger.logger_id:
                refTime = eTime
                break
        # Decide what to do
        midTime = refTime
        toaTimes = []
        for match in matchRecord.matchedEvents:
            logX,logY = match[0].logger.position.xy()
            dist = math.hypot(logX-testX,logY-testY)
            time = dist/self.world.SOUND_SPEED
            estTOA = midTime + time
            toaTimes.append(estTOA)
        toaTimes = numpy.array(toaTimes)
        return toaTimes


    def loglikelihood(self,estimatedTimes,matchRecord):
        # compute the middle time
        # use median as a robust estimator
        upper_quartile = numpy.percentile(estimatedTimes, 75)
        lower_quartile = numpy.percentile(estimatedTimes, 25)
        midTime = (upper_quartile+lower_quartile)/2.0
        llh_array =[]
        # compute llh
        for time,event in zip(estimatedTimes,matchRecord.matchedEvents):
            if (event[0].event_time - midTime) > self.maxTime:
                llh = -100
            else:
                lh = self.likelihood(0,self.timeDev,time-midTime)
                llh = numpy.log10(lh + 1e-50)
            if (event[1].matchConfidence < self.matchThreshold):
                llh = -50
            llh_array.append(llh)
        total_llh = numpy.sum(llh_array)
        stddev = numpy.std(estimatedTimes)
        return total_llh, llh_array

    
    def estimatePositionFromMatch(self,matchRecord,mode='best'):
        # initialize
        bestllh = -1e6
        best_X = 0
        best_Y = 0
        # save it in case we want to plot it etc
        self.llh_record = []
        self.x_record = []
        self.y_record = []
        for trial in range(self.numPoints):
            # pick a point: biased estimator - as we get closer to the end of 
            # the trial, become more likely to pick points in an ever
            # diminishing region around (best_X,best_Y)
            if numpy.random.uniform(0,1) > trial/self.numPoints:
                x = numpy.random.uniform(self.world.extents[0],self.world.extents[1])
                y = numpy.random.uniform(self.world.extents[2],self.world.extents[3])
            else:
                x_dist = self.world.extents[1] - self.world.extents[0]
                y_dist = self.world.extents[3] - self.world.extents[2]
                x = numpy.random.normal(best_X,x_dist*(1.0-trial/self.numPoints))
                y = numpy.random.normal(best_Y,y_dist*(1.0-trial/self.numPoints))
            estimatedEmissionTime = self.estimateTimeOfEmission(matchRecord,x,y)
            llh,llh_array = self.loglikelihood(estimatedEmissionTime,matchRecord)
            self.llh_record.append(llh)
            self.x_record.append(x)
            self.y_record.append(y)
            if llh > bestllh:
                bestllh = llh
                best_llh_array = llh_array
                best_X = x
                best_Y = y
        if mode == 'best':
            return best_X,best_Y,bestllh
        elif mode == 'partial':
            return best_X,best_Y,bestllh,best_llh_array
        elif mode == 'full':
            return self.x_record,self.y_record,self.llh_record
        else:
            raise Exception("Invalid mode string")

    
    def estimateRobustPositionFromMatch(self,matchRecord,minStations=4):
        """ In this code, we throw away the worst (lowest likelihood) time of arrivals """
        initialStations = len(matchRecord.matchedEvents)
        discardCount = 0
        maxDiscard = initialStations - minStations
        if maxDiscard < 1:
            x,y,llh = self.estimatePositionFromMatch(matchRecord)
            self.best_x = x
            self.best_y = y
            self.best_llh = llh
            return x,y,llh

        # start: baseline: all stations:
        x,y,best_llh = self.estimatePositionFromMatch(matchRecord)
        best = [x,y,best_llh]
        localRecord = deepcopy(matchRecord)
        while (discardCount <= maxDiscard):
            x,y,llh,llh_array = self.estimatePositionFromMatch(localRecord,'partial')
            if llh > best_llh:
                best_llh = llh
                best = [x,y,llh]
            # we now decide which events should be removed - we are not allowed to remove
            # the reference event though, or else we end up in a strange situation
            worst_station = numpy.argmin(llh_array)
            if (localRecord.matchedEvents[worst_station][0].logger.logger_id ==
                localRecord.referenceEvent.logger.logger_id):
                llh_array[worst_station] = 1e6
                worst_station = numpy.argmin(llh_array)


            localRecord.matchedEvents.pop(worst_station)
            discardCount += 1
        self.best_x = best[0]
        self.best_y = best[1]
        self.best_llh = best[2]
        return best[0],best[1],best[2]

    def plotLocation(self,numPoints=10,bestPos=False):
        # stations
        x_pos = []
        y_pos = []
        for logger in self.world.loggers:
            x,y = logger.position.xy()
            x_pos.append(x)
            y_pos.append(y)
        label = [p.logger_id for p in self.world.loggers]
        pylab.scatter(x_pos,y_pos,label='stations',facecolor='orange')
        for x,y,l in zip(x_pos,y_pos,label):
            pylab.text(x+40,y,l)
        # points
        if bestPos == True:
            pylab.scatter(self.best_x,self.best_y,c='blue',s=20)
        else:
            sortedidx = numpy.argsort(self.llh_record)
            sortedidx = sortedidx[::-1]
            topk = sortedidx[:numPoints]
            x_points = numpy.array(self.x_record)
            y_points = numpy.array(self.y_record)
            pylab.scatter(x_points[topk],y_points[topk],
                                    c=numpy.linspace(1.0,0.0,numPoints),
                                    s=numpy.linspace(1.0,0.0,numPoints)*20)
        # world bounds
        pylab.xlim(self.world.extents[0],self.world.extents[1])  
        pylab.ylim(self.world.extents[2],self.world.extents[3])
        # gridlines
        pylab.grid(True,which='both')

    def plotTarget(self,target):
        # target
        x,y = target.position.xy()
        pylab.scatter(x,y,label='targets',facecolor='red',marker='*')

        

    
