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
        class Filters:
            k = 20
            std_multiplier = 2.5
        class Volume:
            smooth_sigma = 1.5
            margin_multiplier = 5 # Aumentato leggermente per coprire bene l'impronta sparsa
            taper_px = 3.0          # Pixel di sfumatura per portare a zero il volume sui bordi
            min_thickness = 0.1    # Spessore minimo garantito per compensare la curvatura locale
            spline_smoothing = 0.5  # Regolarizzazione della spline (0 = passa esattamente per i punti, >0 la ammorbidisce) 
        occupancyLimit = 2 #2
        resolution = 0.25
    class General:
        Activation = True
        Maker = True
        Destroyer = False
        Version = 0