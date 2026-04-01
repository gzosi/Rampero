import numpy as np
#%% Defining Config Packet
class Task1:
    class MetaData:
        CavityExt = 'Cavity.pkl'
        CloudExt = 'Cloud.pkl'
    class Settings:
        class Calib:
            Dataset = 'Dataset1'
            Pair = ('Camera1', 'Camera2')
            Model = 'Model27'
        occupancyLimit = 2 #2
        resolution = 0.25
    class General:
        Activation = False
        Maker = True
        Destroyer = False
        Version = 0