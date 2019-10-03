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
    
class AoA_Model():
    
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
    
    @staticmethod
    def llh_numpy(data, mean, sigma):
        s = (data - mean) ** 2 / (2 * (sigma ** 2))
        pdfs = numpy.exp(- s)
        pdfs /= numpy.sqrt(2 * numpy.pi) * sigma
        return numpy.log(pdfs).sum()

    @staticmethod
    def AoA_sanity_filter(delay_mx,max_delay=15):
        """ returns true if it is possible that this delay_mx was generated physically"""
        if numpy.sum(numpy.abs(delay_mx)<max_delay) >= 16:
            return True
        return False
    
    @staticmethod
    def angleToCompass(angle):
        dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        ix = int((angle)/22.5)
        return dirs[ix]
    
    def __init__(self,world,spacing=8.4,meas_dev=1.0):
        """ World: global parameters
            spacing: distance between microphones in cm
            meas_dev: std.deviation of number of samples jitter "
            """
        self.Fs = world.SAMPLE_RATE
        self.c = world.SOUND_SPEED
        self.spacing = spacing
        self.meas_dev = meas_dev
            
    def simulate_delay(self,angle,mic_angle):
        """ Return the fractional delay in samples, relative to origin of array, given the angle of the source and angle of
            microphone"""
        # place a far-field source at an arbitrary distance
        d = 1000.0
        s = self.spacing/2
        # we compensate for the angle of the microphone (essentially a coordinate transformation)
        angle = angle - mic_angle
        x = numpy.sqrt(d**2 - 2*d*s*numpy.cos(angle/180.0*numpy.pi)+s**2)
        # convert a path to a delay. 1000.0 corresponds to no delay
        delta_dist = x - d
        # delay in time
        delta_time = (delta_dist)/self.c
        # delay in samples, remember that spacing is in cm, not m, so need to scale accordingly
        delta_samples = delta_time*self.Fs/100.0
        return delta_samples
    
    def simulate_delay_matrix(self,angle):
        matrix = numpy.zeros((4,4))
        # GS002: mic_angle = [0,90,180,270]
        # GS002b: updated mic_angles to correctly project them with
        # - MIC1 facing North (0 deg)
        # - MIC4 facing East (90 deg)
        # - MIC3 facing South (180 deg)
        # - MIC2 facing West (270 deg)
        mic_angle = [0,270,180,90]
        for j in range(4):
            for k in range(4): 
                delay_j = self.simulate_delay(angle,mic_angle[j])
                delay_k = self.simulate_delay(angle,mic_angle[k])
                delta_delay = delay_k - delay_j
                matrix[j][k] = delta_delay
        return matrix
    
    def predict_measurement(self,loggers,particle):
        if (len(loggers) != 1):
            raise Exception("AoA.predict_measurement expects a single logger in the list")
        angle = numpy.arctan2(particle.x-loggers[0].position.x,particle.y-loggers[0].position.y)/numpy.pi*180.0
        if angle < 0.0:
            angle = angle + 360.0
        mx = self.simulate_delay_matrix(angle)
        return mx
    
    def estimate_log_likelihood(self,observation,loggers,particle):
        dmx = self.predict_measurement(loggers,particle)
        delta = dmx - observation
        llh = self.llh_numpy(numpy.triu(delta),0.0,self.meas_dev)
        return llh

    def estimate_angle(self, delay_mx):
        """ Estimate the angle of arrival in degrees """
        # we now simply run a llh estimator using our model
        # we start with a coarse estimator and then refine with a finer step
        # this trivial change can halve the time to run, which is a big deal for 
        # extraction
        best_angle = None
        best_llh = -1e100
        for angle in range(0,360,20):
            predicted_mx = self.simulate_delay_matrix(angle)
            delta = delay_mx - predicted_mx
            llh = self.llh_numpy(numpy.triu(delta),0.0,self.meas_dev)
            if llh > best_llh:
                best_llh = llh
                best_angle = angle
                #print (best_angle,llh)
        for angle in range(best_angle-10,best_angle+10,1):
            predicted_mx = self.simulate_delay_matrix(angle)
            delta = delay_mx - predicted_mx
            llh = self.llh_numpy(numpy.triu(delta),0.0,self.meas_dev)
            if llh > best_llh:
                best_llh = llh
                best_angle = angle
                #print (best_angle,llh)
        return best_angle,best_llh

    def estimate_angles(self,lags,max_delay=20):
        angles = []
        likelihoods = []
        for lag in lags:
            if self.AoA_sanity_filter(lag,max_delay):
                angle,llh = self.estimate_angle(lag)
                angles.append(angle)
                likelihoods.append(llh)
            else:
                angles.append(None)
                likelihoods.append(None)
        return angles,likelihoods
    
class TDoA_Model():
    
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
    
    def __init__(self,world,meas_dev=1000):
        self.world = world
        self.meas_dev = meas_dev 
    
    def predict_measurement(self,loggers,particle):
        l0x,l0y = loggers[0].position.xy()
        l1x,l1y = loggers[1].position.xy()
        if (len(loggers) != 2):
            raise Exception("TDOA.predict_measurement expects a pair of loggers")
        dist_l1_p = numpy.sqrt((particle.x - l0x)**2 +\
                            (particle.y - l0y)**2)
        dist_l2_p = numpy.sqrt((particle.x - l1x)**2 +\
                            (particle.y - l1y)**2)
        delta_dist = dist_l1_p - dist_l2_p
        delta_samples = delta_dist/self.world.SOUND_SPEED*self.world.SAMPLE_RATE
        return delta_samples
    
    def estimate_log_likelihood(self,observation,loggers,particle):
        particle_delta_distance = self.predict_measurement(loggers,particle)
        l = self.likelihood(particle_delta_distance,\
                                        self.meas_dev,\
                                        observation)
        return numpy.log10(l+1e-30)