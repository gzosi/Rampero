#%% Defining Config Packet
class Task3:
    class MetaData:
        OutputExt = 'Data.pkl'
        MeshExt = 'Mesh.ply'
    class Settings:
        class Calib:
            Dataset = 'Dataset1'
            Pair = ('Camera1', 'Camera2')
            Model = 'Model27'
        Bounds = [ [64, 84], [128, 148], [192, 212]]
        class LocalFilter:
            class Size:
                voxelSize = 0.5
            class Radius:
                nbPoints = 25
                radius = 5.0
            class Stats:
                nbNeighbors = 100
                stdRatio = 0.25
        class GlobalFilter:
            class Positional:
                enabled = True
                exclusion_zones = [ 
                    [[-100, -100 , -1000], [-18, -5, 1000]],
                    [[50, -1000, - 1000],[100, 1000, 1000]],
                    [[-100, -10, 475],[-8, 10, 1000]]
                ]
            class Size:
                voxelSize = 2.5        
            class Stats:
                enabled = True
                nbNeighbors = 30
                stdRatio = 1.0
            class Radius:
                enabled = True
                nbPoints = 15           
                radius = 5.0
            class PCA:
                enabled = True
                search_radius = 10      
                max_nn = 50
                threshold = 0.15  
        class Mesh:
            class NormalOrient:
                radius = 25           
                max_nn = 150              
                cameraLoc = [0.0, 0.0, 0.0]
            class Poisson:
                depth = 9           
                width = 0.0
                scale = 1.1          
                linear_fit = True    
                density_trim = 0.05
            class Cleaning:
                max_distance_from_source = 3
                max_triangle_edge = 25       
            class Smoothing:
                subdivide = 2                         
                method = 'taubin'        
                iterations = 500       
                lambda_filter = 0.5
                mu = -0.53
    class General:
        Activation = True
        Maker = True
        Destroyer = False
        Version = 0