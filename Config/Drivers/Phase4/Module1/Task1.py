import numpy as np

#%% Defining Config Packet
class Task1:
    class MetaData:
        OutputExt = 'Data.pkl'
    class Settings:
        class Src:
            Database = 'Database4'
        class Ref:
            Database = 'Database3'
            Dataset = 'Dataset2'
            Location = 'Background'
        Bounds = [[10, 10]] # [[8, 12]]
        class SmartPrompt:
            ApplyCLAHE = True      # Fondamentale per far risaltare bolle trasparenti (Img 3)
            TextureKernel = 15     # Dimensione finestra varianza. Aumenta per nuvole grandi.
            TextureThreshold = 25  # ALZATO: Più selettivo sul rumore di fondo (prima 20)
            MorphKernel = (5, 5)   # Kernel per pulire la maschera binaria di tentativo
            BoxPadding = 10        # RIDOTTO: Box più stretta intorno alla bolla individuata (prima 15)
            AreaMin = 10          # ALZATO: Scarta macchie di rumore più piccole (prima 50)
        class DynamicROI:
            Camera1 = {
                8 : np.array([
                    [748, 392], [564, 412], [376, 356], 
                    [425, 266], [557, 302], [711, 318]]),
                10 : np.array([
                    [729, 360], [568, 398], [390, 349], 
                    [419, 262], [554, 303], [689, 292]]),
                12 : np.array([
                    [725, 349], [593, 409], [389, 346], 
                    [412, 264], [571, 302], [696, 291]]),
            }
            Camera2 = {
                8 : np.array([
                    [462, 269], [319, 311], [154, 254], 
                    [179, 180], [312, 212], [441, 213]]),
                10 : np.array([
                    [453, 273], [326, 312], [157, 264], 
                    [180, 179], [317, 212], [414, 200]]),
                12 : np.array([
                    [446, 281], [315, 317], [150, 262], 
                    [169, 170], [317, 210], [414, 211]]),
            }
        class Difference:
            BlurKsize = (5, 5)
            ClaheClipLimit = 2.0
            ClaheGridSize = (8, 8)
            ThreshVal = 30
            ThreshMax = 255
        class Segmenter: 
            Model = 'SamHq'
            Checkpoint = 'sam_hq_vit_h.pth'
            Name = 'vit_h'
    class General:
        Activation = True
        Maker = True
        Destroyer = False
        Version = 0