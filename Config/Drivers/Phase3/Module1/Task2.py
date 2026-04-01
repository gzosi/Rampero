class Task2:
    class MetaData:
        OutputName = 'Data'
        OutputExt = '.pkl'
    class Settings:
        class Src:
            Database = 'Database3'
            Dataset = 'Dataset2'
            Foreground = 'Foreground'
        class Ref:
            Database = 'Database1'
            Dataset = 'Dataset1'
            Record = 'Record1'
        class Calib:
            Dataset = 'Dataset1'
            Pair = ('Camera1', 'Camera2')
            Model = 'Model27'
        EpiLimit = 25
        ConfLimit = 0
        SimLimit = 0.5
    class General:
        Activation = False
        Maker = True
        Destroyer = False
        Version = 0