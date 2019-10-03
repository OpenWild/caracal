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

### Core library functions and classes including:
# Pickler: utility for saving and restoring python data
# Position: generic position class, supports LatLong, UTM and XY coordinate types
# Logger: logger object and associated directory of data
# Target: ground truth event at a certain location and/or time
# CoordinateTransform: helper function to convert LatLong to cartesian
# World: container for a collection of loggers and targets
# 


import pickle
import numpy
import pyproj
import pylab
import os


class Pickler():
    
    def __init__(self,parent_directory):
        self.parent_directory = parent_directory
        
    
    def save(self,data,filename,sub_dir=None):
        if sub_dir is None:
            full_filename = os.path.join(self.parent_directory,filename)
        else:
            # check that the subdir exists, else create it
            full_dir = os.path.join(self.parent_directory,sub_dir)
            if not (os.path.isdir(full_dir)):
                os.makedirs(full_dir)
            full_filename = os.path.join(full_dir,filename)
            with (open(full_filename,'wb')) as f:
                pickle.dump(data,f)
                
    def load(self,filename,sub_dir=None):
        if sub_dir is None:
            full_filename = os.path.join(self.parent_directory,filename)
        else:
            full_dir = os.path.join(self.parent_directory,sub_dir)
            full_filename = os.path.join(full_dir,filename)
        with (open(full_filename,'rb')) as f:
                data = pickle.load(f)
        return data

class Position():
    def __init__(self,coord_h,coord_v,positiontype):
        """positiontype sets the type of position. Can be:
        - 'LatLong'
        - 'UTM'
        - 'XY'
        """
        self.h = coord_h
        self.v = coord_v
        self.positiontype = positiontype
        # A reference position acts as the origin
        self.origin = None
        if positiontype == 'XY':
            self.x = self.h
            self.y = self.v
        elif positiontype  == 'UTM':
            self.x = self.h
            self.y = self.v
        elif positiontype == 'LatLong':
            c = CoordinateTransform()
            utm_x,utm_y = c.transform(self.h,self.v)
            self.x = utm_x
            self.y = utm_y
        else:
            raise Exception("Invalid positiontype")

    
    def setOrigin(self,origin):
        self.origin = origin
        
    def xy(self):
        """Return coordinates in origin referenced XY format"""
        if self.origin is not None:
            return self.x - self.origin.x,self.y - self.origin.y
        else:
            return self.x,self.y
        
    def distance(self,position_b):
        """Distance to another position"""
        delta_x = self.x - position_b.x
        delta_y = self.y - position_b.y
        dist = numpy.sqrt(delta_x**2+delta_y**2)
        return dist
    
    def __repr__(self):
        x,y = self.xy()
        return "Position (" + str(x) + ":" + str(y) + ")" 
    
class Logger():
    
    def __init__(self,id,name,position,directory,filetemplate):
        """ This is a container which holds all the details of a particular logger"""
        self.logger_id = id
        self.logger_name = name
        # generic position instance
        self.position = position
        # where to find all the files
        self.directory = directory
        self.filetemplate = filetemplate
        
    def __repr__(self):
        return "Logger " + str(self.logger_id) + ":" + self.logger_name + ")" 

class Target():
    
    def __init__(self,target_id,target_name,position,target_timestamp=None):
        """ This is a container which holds all the details of a particular target event.
        A target is a surveyed (ground-truthed) event."""
        self.target_id = target_id
        self.target_name = target_name
        # generic position instance
        self.position = position
        # timestamp (if known) when event occured
        self.target_timestamp = target_timestamp
        
    def __repr__(self):
        return "Target " + str(self.target_id) + ":" + self.target_name +\
                " Pos: (" + str(self.position.x) + ";" + str(self.position.y) + ")" +\
                " Timestamp: " + str(self.target_timestamp)
            
class CoordinateTransform():
    
    def __init__(self):
        """ This class deals with coordinate transformations"""
        self._projections = {}

    def zone(self,coordinates):
        if 56 <= coordinates[1] < 64 and 3 <= coordinates[0] < 12:
            return 32
        if 72 <= coordinates[1] < 84 and 0 <= coordinates[0] < 42:
            if coordinates[0] < 9:
                return 31
            elif coordinates[0] < 21:
                return 33
            elif coordinates[0] < 33:
                return 35
            return 37
        return int((coordinates[0] + 180) / 6) + 1


    def letter(self,coordinates):
        return 'CDEFGHJKLMNPQRSTUVWXX'[int((coordinates[1] + 80) / 8)]


    def project(self,coordinates):
        z = self.zone(coordinates)
        l = self.letter(coordinates)
        #print(z,l)
        if z not in self._projections:
            self._projections[z] = pyproj.Proj(proj='utm', zone=z, ellps='WGS84')
        x, y = self._projections[z](coordinates[0], coordinates[1])
        if y < 0:
            y += 10000000
        return z, l, x, y


    def unproject(self,z, l, x, y):
        if z not in self._projections:
            self._projections[z] = pyproj.Proj(proj='utm', zone=z, ellps='WGS84')
        if l < 'N':
            y -= 10000000
        lng, lat = self._projections[z](x, y, inverse=True)
        return (lng, lat)

    def transform(self,lat,long):
        z,l,x,y = self.project((long,lat))
        return x,y

class World():
    
    def _determineOrigin(self):
        if self.definedOrigin is None:
            # Internal method which is called whenever a new
            # logger or target is added to determine if a new
            # origin is needed
            min_x = 1e100
            min_y = 1e100
            # First find the origin
            for l in self.loggers:
                if (l.position.x < min_x):
                    min_x = l.position.x
                if (l.position.y < min_y):
                    min_y = l.position.y
            for t in self.targets:
                if (t.position.x < min_x):
                    min_x = t.position.x
                if (t.position.y < min_y):
                    min_y = t.position.y
            # Create the origin
            origin = Position(min_x,min_y,"XY")
            # Update everyone
            for l in self.loggers:
                l.position.setOrigin(origin)
            for t in self.targets:
                t.position.setOrigin(origin)
        else:
            for l in self.loggers:
                l.position.setOrigin(self.definedOrigin)
            for t in self.targets:
                t.position.setOrigin(self.definedOrigin)


    
    def __init__(self,extents,sound_speed=343,sample_rate=44100):
        self.extents = extents
        self.loggers = []
        self.targets = []
        self.SOUND_SPEED = sound_speed  
        self.SAMPLE_RATE = sample_rate
        self.definedOrigin = None
        
    def addLogger(self,logger):
        self.loggers.append(logger)
        self._determineOrigin()
        
    def addTarget(self,target):
        self.targets.append(target)
        self._determineOrigin()

    def setOrigin(self,item):
        # Assume we are given a target/logger
        self.definedOrigin = item.position
        self._determineOrigin()
        
    def position_generator(self):
        # ToDo: This should get moved somewhere else, it does not really belong here
        """ Generate a random position from the world for a particle/candidate solution"""
        return [numpy.random.uniform(self.extents[0],self.extents[1]),
                numpy.random.uniform(self.extents[2],self.extents[3])]
              
    def __repr__(self):
        ret_str = ''
        ret_str += '\nLoggers:\n'
        for l in self.loggers:
            ret_str += str(l) + '\n'
        ret_str += '\nTargets:\n'
        for t in self.targets:
            ret_str += str(t) + '\n'
        return ret_str