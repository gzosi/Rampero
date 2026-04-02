import numpy as np
#%% Defining Config Packet
class Task1:
    class MetaData:
        OutputName = 'Data'
        OutputExt = '.pkl'
    class Settings:
        class Src:
            Database = 'Database3'
            Dataset = 'Dataset4'
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
                    [798, 164], [616, 318], [455, 429], 
                    [392, 334], [534, 161], [799, 30]]),
                10 : np.array([
                    [855, 218], [679, 362], [427, 476], 
                    [403, 350], [576, 179], [832, 68]]),
                20 : np.array([
                    [846, 356], [637, 484], [396, 544], 
                    [353, 354], [639, 194], [842, 135]]),
                30 : np.array([
                    [837, 473], [609, 560], [337, 579], 
                    [323, 443], [592, 306], [811, 234]]),
                40 : np.array([
                    [828, 534], [548, 617], [355, 609], 
                    [336, 505], [559, 387], [791, 282]]),
                50 : np.array([
                    [771, 641], [556, 677], [330, 631], 
                    [332, 532], [574, 455], [736, 387]]),
            }
            Camera2 = {
                0 : np.array([
                    [800, 249], [642, 399], [445, 503], 
                    [428, 383], [586, 245], [805, 129]]),
                10 : np.array([
                    [887, 292], [695, 473], [471, 577], 
                    [441, 439], [646, 261], [872, 161]]),
                20 : np.array([
                    [880, 455], [699, 577], [442, 624], 
                    [440, 479], [654, 348], [867, 250]]),
                30 : np.array([
                    [875, 559], [698, 655], [433, 656], 
                    [423, 517], [660, 373], [870, 301]]),
                40 : np.array([
                    [876, 673], [684, 690], [422, 671], 
                    [417, 593], [640, 457], [874, 393]]),
                50 : np.array([
                    [883, 686], [678, 691], [440, 690], 
                    [471, 624], [682, 519], [875, 482]]),
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