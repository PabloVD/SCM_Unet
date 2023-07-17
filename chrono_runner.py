import os
import numpy as np

# Uncomment to allow reproducibility
np.random.seed(seed=1234)

ini_sim = 0
Nsims = 10

path_hmaps = "/home/tda/CARLA/TrainChrono/SCM_ForceGNN/HeightMaps/hmaps_slope/"
tend = 5

out_path = "/home/tda/Descargas/SCM_simulations/NoProcessed"

use_nn = 0

simtype = "nn" if use_nn else "scm"

for sim in range(ini_sim,Nsims):

    print("Simulation "+str(sim))

    hmap_path = path_hmaps+"hmap_"+str(sim)+".png"
    throttle = np.random.uniform(0.6,1.)
    steering = np.random.uniform(-1.,1.)
    initheight = np.random.uniform(0.,0.15)

    print("throttle: {:.2f}".format(throttle)+", steering: {:.2f}".format(steering))

    commandline = "./test_Polaris_SCM_CustomTerrain"
    commandline += " --terrain_dir=" + hmap_path
    commandline += " --tend="+str(tend)
    commandline += " --throttle={:.2f}".format(throttle)
    commandline += " --steering={:.2f}".format(steering)
    commandline += " --initheight={:.2f}".format(initheight)
    commandline += " --use_nn="+str(use_nn)
    commandline += " > out_"+simtype+"_"+str(sim)+".txt"
    
    
    print(commandline)
    #exit()

    os.system(commandline)
    os.system("mv DEMO_OUTPUT/POLARIS_SCM_* " +out_path)

