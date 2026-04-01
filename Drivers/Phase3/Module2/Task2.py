#%% Importing Libreries
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from termcolor import colored
#%% Defining Main Function
def main(Config):       
    task_conf = Config.Packages.Drivers.Phases.Phase3.Modules.Module2.Tasks.Task2
    if task_conf.General.Activation is True:
        print('.... Task2:', colored('Running ℹ️', 'cyan'))
        try:
            main_root = Path(Config.Paths.mainRooot)
            srcRoot = (
                main_root /
                Config.Paths.DataRoots.ResourcesRoot /
                Config.Paths.DataRoots.StreamRoot / 
                Config.Paths.DataRoots.CaseStudyRoot() /
                Config.Packages.Drivers.__name__ / 
                Config.Packages.Drivers.Phases.Phase3.__name__ / 
                Config.Packages.Drivers.Phases.Phase3.Modules.Module2.__name__ /
                Config.Packages.Drivers.Phases.Phase3.Modules.Module2.Tasks.Task1.__name__ )
            dstRoot = (
                main_root /
                Config.Paths.DataRoots.ResourcesRoot /
                Config.Paths.DataRoots.StreamRoot / 
                Config.Paths.DataRoots.CaseStudyRoot() /
                Config.Packages.Drivers.__name__ / 
                Config.Packages.Drivers.Phases.Phase3.__name__ / 
                Config.Packages.Drivers.Phases.Phase3.Modules.Module2.__name__ /
                task_conf.__name__ /
                task_conf.MetaData.OutputExt)
            cavityData = pd.read_pickle((srcRoot / Config.Packages.Drivers.Phases.Phase3.Modules.Module2.Tasks.Task1.MetaData.CavityExt))
            cloudData = pd.read_pickle((srcRoot / Config.Packages.Drivers.Phases.Phase3.Modules.Module2.Tasks.Task1.MetaData.CloudExt))
            period = Config.Settings.Acquisition.PPR 
            blades = Config.Settings.Acquisition.Blades
            rows = []
            for (cavityKey, cavity), (cloudKey, cloud) in zip(cavityData.items(), cloudData.items()):
                if cavity.empty: continue
                row = {
                    'key': cavityKey,
                    'Phase': int(cavityKey) % period,
                    'Blade': (int(cavityKey) % period) % (blades+1)}
                for col in ['Area', 'Volume']:
                    row[col] = [cavity[col].sum(), cloud[col].sum()]
                for col in ['Inner', 'Outer', 'Control']:
                    cavity_vals = cavity[col].dropna().values
                    cloud_vals = cloud[col].dropna().values
                    row[col] = [
                        np.concatenate(cavity_vals) if len(cavity_vals) > 0 else np.array([]),
                        np.concatenate(cloud_vals) if len(cloud_vals) > 0 else np.array([])]
                rows.append(row)
            aggregate = pd.DataFrame(rows)  
            pickle.dump(aggregate, open(dstRoot, 'wb'))
        except Exception as e:
            print('.... Task2:', colored(f'Error: {e} ❌', 'red'))
            raise e
    elif task_conf.General.Activation is False:
        print('.... Task2:', colored('Offline ⚠️', 'yellow'))
    else:
        raise ValueError('Please Set the Task2 Switch (on/off) ❌')
    return