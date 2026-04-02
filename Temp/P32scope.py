#%% Importing Libraries
from pathlib import Path
import pandas as pd
import pyvista as pv
import numpy as np

from Config import Config

# =============================================================================
# MODIFICA QUESTA VARIABILE PER SCEGLIERE QUALE ELEMENTO VEDERE
INDICE_ELEMENTO = 451  
# =============================================================================

# =============================================================================
# CONFIGURAZIONE PERCORSI
# =============================================================================
task_conf = Config.Packages.Drivers.Phases.Phase3.Modules.Module2.Tasks.Task1
main_root = Path(Config.Paths.mainRooot)

CAVITY_DIR = (main_root /
    Config.Paths.DataRoots.ResourcesRoot /
    Config.Paths.DataRoots.StreamRoot / 
    Config.Paths.DataRoots.CaseStudyRoot() /
    Config.Packages.Drivers.__name__ /
    Config.Packages.Drivers.Phases.Phase3.__name__ /
    Config.Packages.Drivers.Phases.Phase3.Modules.Module2.__name__ /
    task_conf.__name__ /
    task_conf.MetaData.CavityExt)

CLOUD_DIR = (main_root /
    Config.Paths.DataRoots.ResourcesRoot /
    Config.Paths.DataRoots.StreamRoot / 
    Config.Paths.DataRoots.CaseStudyRoot() /
    Config.Packages.Drivers.__name__ /
    Config.Packages.Drivers.Phases.Phase3.__name__ /
    Config.Packages.Drivers.Phases.Phase3.Modules.Module2.__name__ /
    task_conf.__name__ /
    task_conf.MetaData.CloudExt)

POSE_DIR = (main_root /
    Config.Paths.DataRoots.ResourcesRoot /
    Config.Paths.DataRoots.StreamRoot / 
    Config.Paths.DataRoots.CaseStudyRoot() /
    Config.Packages.Drivers.__name__ /
    Config.Packages.Drivers.Phases.Phase2.__name__ /
    Config.Packages.Drivers.Phases.Phase2.Modules.Module2.__name__ /
    Config.Packages.Drivers.Phases.Phase2.Modules.Module2.Tasks.Task4.__name__ /
    Config.Packages.Drivers.Phases.Phase2.Modules.Module2.Tasks.Task4.MetaData.OutputExt)

# ---------------------------------------------------------

# ---------------------------------------------------------
print("Caricamento dati in corso...")
POSE = pd.read_pickle(POSE_DIR)['points']
CAVITY = pd.read_pickle(CAVITY_DIR)
CLOUD = pd.read_pickle(CLOUD_DIR)

# =============================================================================
# PARAMETRI VISIVI
# =============================================================================
ALPHA = 0.99         # Trasparenza della mesh
MAX_SCREEN_W = 1600  # Larghezza massima della finestra
MAX_SCREEN_H = 900   # Altezza massima della finestra

# =============================================================================
# FUNZIONI DI VISUALIZZAZIONE
# =============================================================================

def visualize_side_by_side(pose_pts, cavity_df, cloud_df, element_name):
    """
    Visualizza in 3D i risultati affiancati per Cavity e Cloud, 
    iterando su tutte le righe dei DataFrame forniti.
    
    Parametri:
    - pose_pts: numpy array dei punti base (nuvola di riferimento).
    - cavity_df: DataFrame di Pandas con i dati Cavity per l'elemento scelto.
    - cloud_df: DataFrame di Pandas con i dati Cloud per l'elemento scelto.
    - element_name: Stringa, il nome della chiave del dizionario (es. '00000').
    """
    # Creiamo un plotter con 1 riga e 2 colonne
    plotter = pv.Plotter(shape=(1, 2), window_size=[MAX_SCREEN_W, MAX_SCREEN_H])
    
    def _plot_dataframe(plotter, title, pose, df):
        # 0. Punti Pose (Sfondo/Riferimento)
        if pose is not None and len(pose) > 0:
            plotter.add_points(
                pose, 
                color='darkgray', 
                point_size=2.0,
                label='Pose Base'
            )

        tot_area = 0.0
        tot_volume = 0.0

        # Cicliamo su tutte le righe del DataFrame
        for i, (index, row) in enumerate(df.iterrows()):
            inner = row.get('Inner')
            outer = row.get('Outer')
            control = row.get('Control')
            mesh = row.get('Mesh')
            
            # Sommiamo area e volume gestendo eventuali NaN
            area = row.get('Area', 0)
            volume = row.get('Volume', 0)
            if pd.notna(area): tot_area += area
            if pd.notna(volume): tot_volume += volume

            # Per evitare di riempire la legenda con duplicati, diamo la label solo alla prima riga plottata
            lbl_inner = 'Inner' if i == 0 else None
            lbl_outer = 'Outer' if i == 0 else None
            lbl_control = 'Control' if i == 0 else None
            lbl_mesh = 'Volume Mesh' if i == 0 else None

            # 1. Punti Inner
            if isinstance(inner, np.ndarray) and len(inner) > 0:
                plotter.add_points(inner, color='royalblue', point_size=3.0, render_points_as_spheres=True, label=lbl_inner)

            # 2. Punti Control
            if isinstance(control, np.ndarray) and len(control) > 0:
                plotter.add_points(control, color='darkblue', point_size=6.0, render_points_as_spheres=True, label=lbl_control)

            # 3. Punti Outer
            if isinstance(outer, np.ndarray) and len(outer) > 0:
                plotter.add_points(outer, color='royalblue', point_size=3.0, render_points_as_spheres=True, label=lbl_outer)

            # 4. Volume Mesh
            if mesh is not None:
                # Assumiamo che 'mesh' sia già un oggetto pyvista (es. UnstructuredGrid)
                plotter.add_mesh(mesh, color='lightblue', opacity=ALPHA, show_edges=True, edge_color='lightblue', label=lbl_mesh)

        # Aggiunta dei Titoli e Testo
        plotter.add_text(title, position='upper_left', font_size=14)
        
        # Testo Statistiche Complessive in alto a destra
        stats_text = f"Elementi (Righe): {len(df)}\nArea Totale: {tot_area:.2f}\nVolume Totale: {tot_volume:.2f}"
        plotter.add_text(stats_text, position='upper_right', font_size=10, color='white')

        # Impostazioni vista
        plotter.add_legend(size=(0.2, 0.2))
        plotter.show_grid()
        plotter.show_axes()

    # --- Sottofinestra 1 (Sinistra): Cavity Data ---
    plotter.subplot(0, 0)
    _plot_dataframe(plotter, f"Cavity Data [{element_name}]", pose_pts, cavity_df)
    
    # --- Sottofinestra 2 (Destra): Cloud Data ---
    plotter.subplot(0, 1)
    _plot_dataframe(plotter, f"Cloud Data [{element_name}]", pose_pts, cloud_df)

    # Colleghiamo le telecamere in modo che ruotando una si ruoti anche l'altra
    plotter.link_views()

    # Mostra la finestra interattiva
    plotter.show()


# =============================================================================
# ESECUZIONE
# =============================================================================

if __name__ == "__main__":
    # Assumiamo che le variabili CAVITY, CLOUD e POSE siano caricate in memoria.
    # Estraggo la chiave usando l'indice posizionale da CAVITY
    chiavi_disponibili = list(CAVITY.keys())
    
    if INDICE_ELEMENTO < len(chiavi_disponibili):
        chiave_selezionata = chiavi_disponibili[INDICE_ELEMENTO]
        print(f"Preparazione visualizzazione per l'elemento: {chiave_selezionata}")
        
        # Estrazione DataFrames
        df_cavity_corrente = CAVITY[chiave_selezionata]
        df_cloud_corrente = CLOUD[chiave_selezionata]
        
        # Estrazione POSE (modulo PPR sul valore reale della chiave)
        ppr = Config.Settings.Acquisition.PPR
        
        # Convertiamo la stringa della chiave (es. '00276') in un intero (276)
        numero_elemento_reale = int(chiave_selezionata)
        
        # Applichiamo il modulo per gestire la periodicità della superficie
        indice_pose = numero_elemento_reale % ppr
        
        if isinstance(POSE, dict):
            # Usiamo l'indice_pose posizionale sui valori del dizionario
            pose_corrente = list(POSE.values())[indice_pose]
        else:
            # Nel caso in cui POSE fosse una lista di array
            pose_corrente = POSE[indice_pose]
            
        # Avvio la visualizzazione
        visualize_side_by_side(pose_corrente, df_cavity_corrente, df_cloud_corrente, chiave_selezionata)
    else:
        print(f"ERRORE: L'indice {INDICE_ELEMENTO} è fuori dal range. Ci sono solo {len(chiavi_disponibili)} elementi.")