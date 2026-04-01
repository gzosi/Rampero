import numpy as np
#%% Defining Config Packet
class Task1:
    class MetaData:
        OutputName = 'Data'
        OutputExt = '.pkl'
    class Settings:
        class Src:
            Database = 'Database3'
            Dataset = 'Dataset2'
            Foreground = 'Foreground'
            Background = 'Background'
        Bounds = [[0, 37]] 
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
                    [751, 392], [615, 437], [424, 391], 
                    [358, 261], [530, 244], [741, 307]]),
                10 : np.array([
                    [741, 409], [590, 445], [403, 437], 
                    [300, 266], [471, 230], [711, 331]]),
                20 : np.array([
                    [644, 427], [522, 473], [324, 480], 
                    [241, 314], [444, 280], [625, 338]]),
                30 : np.array([
                    [535, 424], [350, 481], [277, 404], 
                    [267, 299], [387, 271], [519, 312]]),
                40 : np.array([
                    [460, 420], [371, 492], [284, 433], 
                    [266, 324], [364, 302], [431, 336]]),
            }
            Camera2 = {
                0 : np.array([
                    [532, 297], [352, 369], [183, 352], 
                    [109, 189], [306, 153], [490, 209]]),
                10 : np.array([
                    [407, 312], [275, 353], [130, 338], 
                    [86, 184], [244, 192], [391, 235]]),
                20 : np.array([
                    [338, 297], [188, 380], [88, 399], 
                    [6, 209], [180, 177], [318, 214]]),
                30 : np.array([
                    [253, 340], [187, 375], [21, 385], 
                    [8, 234], [102, 179], [237, 239]]),
                40 : np.array([
                    [217, 349], [126, 386], [40, 390], 
                    [11, 230], [94, 215], [156, 246]]),
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