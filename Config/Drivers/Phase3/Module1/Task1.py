import numpy as np
#%% Defining Config Packet
class Task1:
    class MetaData:
        OutputName = 'Data'
        OutputExt = '.pkl'
    class Settings:
        class Src:
            Database = 'Database3'
            Dataset = 'Dataset1'
            Foreground = 'Foreground'
            Background = 'Background'
        Bounds = [[0, 45]] 
        class SmartPrompt:
            ApplyCLAHE = True      # Fondamentale per far risaltare bolle trasparenti (Img 3)
            TextureKernel = 15     # Dimensione finestra varianza. Aumenta per nuvole grandi.
            TextureThreshold = 25  # ALZATO: Più selettivo sul rumore di fondo (prima 20)
            MorphKernel = (5, 5)   # Kernel per pulire la maschera binaria di tentativo
            BoxPadding = 10        # RIDOTTO: Box più stretta intorno alla bolla individuata (prima 15)
            AreaMin = 100          # ALZATO: Scarta macchie di rumore più piccole (prima 50)
        class Cloud:
            BlurKernel = [9, 9]     # Sfocatura per omogeneizzare la texture della schiuma
            Relaxation = 0.85       # Rilassamento soglia Otsu (es. 0.85 = abbassa del 15% per catturare i bordi)
            DilateKernel = [11, 11] # Dimensione per espandere la "search region" di SAM
            DilateIter = 3          # Iterazioni di espansione (aumenta per far crescere di più la nuvola)
        class DynamicROI:
            Camera1 = {
                0 : np.array([
                    [444, 479], [253, 344], [142, 192], 
                    [162, 97], [355, 208], [479, 379]]),
                10 : np.array([
                    [417, 464], [256, 371], [119, 242], 
                    [130, 172], [316, 250], [448, 377]]),
                20 : np.array([
                    [409, 505], [224, 415], [117, 332], 
                    [124, 231], [289, 289], [426, 417]]),
                30 : np.array([
                    [431, 510], [253, 482], [129, 398], 
                    [137, 319], [298, 351], [454, 437]]),
                40 : np.array([
                    [463, 563], [293, 582], [134, 518], 
                    [142, 389], [301, 473], [441, 495]]),
            }

            Camera2 = {
                0 : np.array([
                    [463, 609], [323, 490], [235, 349], 
                    [269, 296], [426, 405], [511, 541]]),
                10 : np.array([
                    [428, 569], [291, 508], [191, 392], 
                    [195, 343], [359, 405], [456, 495]]),
                20 : np.array([
                    [431, 601], [311, 582], [183, 500], 
                    [182, 393], [364, 471], [460, 519]]),
                30 : np.array([
                    [475, 652], [337, 645], [208, 579], 
                    [193, 470], [358, 518], [497, 582]]),
                40 : np.array([
                    [456, 674], [329, 680], [200, 634], 
                    [195, 534], [321, 565], [445, 633]]),
            }
        class Segmenter: 
            Model = 'SamHq'
            Checkpoint = 'sam_hq_vit_h.pth'
            Name = 'vit_h'
        class Collapse:
            Weights = [0.5, 0.5, 0.25] # Area, Media, Std
            Percentile = 9
        class Group:
            Contained = 0.8
            Smaller = 0.75
            Similar = 0.9
    class General:
        Activation = True
        Maker = True
        Destroyer = False
        Version = 0