import numpy as np
#%% Defining Config Packet
class Task1:
    class MetaData:
        CavityExt = 'Cavity.pkl'
        CloudExt = 'Cloud.pkl'
    class Settings:
        class Calib:
            Dataset = 'Dataset4'
            Pair = ('Camera1', 'Camera2')
            Model = 'Model27'
        class Filters:
            k = 10
            std_multiplier = 1.5
        occupancyLimit = 2 #2
        resolution = 0.25
    class General:
        Activation = True
        Maker = True
        Destroyer = False
        Version = 0