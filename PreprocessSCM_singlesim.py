import os
import numpy as np
import pandas as pd
from natsort import natsorted
from tqdm import tqdm
import glob
import argparse


# Get wheel information (dissmiss vehicle info for now)
def wheelinfo(vehiclefile):

    lines = []
    count = 0
    for line in open(vehiclefile):

        parts = line.strip().split(',')
        parts = parts[:-1]  # remove last item, which is just ''
        #print(vehiclefile)
        #print(parts)
        parts = [float(part) for part in parts]
        lines.append(parts)
        count+=1

    #vehicleinfpre = lines[0:2]
    #vehicleinf = np.concatenate( [vehicleinfpre[0], vehicleinfpre[1]] )
    vehicleinf = np.array(lines[0])
    wheelinf = np.array(lines[1:5])
    wheelinf = np.concatenate([wheelinf,np.array(lines[5:])],axis=1)
    #print(vehicleinf.shape, wheelinf.shape)
    #print(vehicleinf, wheelinf)
    #exit()
    return wheelinf, vehicleinf

def find_bounding_box(datasetname, maxtimesteps = None, margin = 0.5):

    if maxtimesteps is None:
        maxtimesteps = (len(glob.glob(datasetname+"/*")) - 1)//2

    wheel_ini, _ = wheelinfo(datasetname+"/mbs_{:d}_.csv".format(0))
    wheel_mid, _ = wheelinfo(datasetname+"/mbs_{:d}_.csv".format((maxtimesteps-1)//2))
    wheel_end, _ = wheelinfo(datasetname+"/mbs_{:d}_.csv".format(maxtimesteps-1))

    pos_ini, pos_mid, pos_end = wheel_ini[:,:3], wheel_mid[:,:3], wheel_end[:,:3]
    totpos = np.concatenate((pos_ini, pos_mid, pos_end), axis=0)

    min_x, max_x = totpos[:,0].min() - margin, totpos[:,0].max() + margin
    min_y, max_y = totpos[:,1].min() - margin, totpos[:,1].max() + margin

    return min_x, max_x, min_y, max_y


def cond_bbox(arr, min_x, max_x, min_y, max_y):

    cond_x = np.logical_and((arr[:,0] > min_x) , (arr[:,0] < max_x) )
    cond_y = np.logical_and((arr[:,1] > min_y) , (arr[:,1] < max_y) )
    cond = np.logical_and(cond_x, cond_y)

    return cond

# Save info for a given burst and wheel
def save_burst(datasetname, outpath, maxtimesteps=None, lims=None):

    if maxtimesteps is None:
        maxtimesteps = (len(glob.glob(datasetname+"/*")) - 1)//2

    x = []
    wheelarr = []
    vehicl = []

    if lims is None:
        min_x, max_x, min_y, max_y = find_bounding_box(datasetname, maxtimesteps)
    else:
        min_x, max_x, min_y, max_y = lims
    #min_x, max_x, min_y, max_y = -34.2153, 28.9142, -1.116, 4.24686
    print("Bounding box:", min_x, max_x, min_y, max_y)
    #exit()

    pbar = tqdm(range(maxtimesteps), total=maxtimesteps, position=0, leave=True, desc=f"Time steps")

    for t in pbar:

        # Soil file (beware with the space after .csv)
        soilfile = datasetname+"/vertices_{:d}.csv".format(t)

        # Wheel info
        wheel, vehicle = wheelinfo(datasetname+"/mbs_{:d}_.csv".format(t))
    
        #wheel = wheel[w]
        wheelarr.append(wheel)
        vehicl.append(vehicle)

        # If soil file is not empty, load it
        

        # Get positions
        if t==0:
            df_x = pd.read_csv(soilfile,delimiter=",",dtype = {'#index': int, 'x': float, 'y': float, 'z': float})
            inds, arr = df_x.values[:,0], df_x.values[:,1:4]
            print(df_x.values.shape)
            #print(arr[:,0].min(), arr[:,0].max(), arr[:,1].min(), arr[:,1].max())
            #exit()
            condbbox = cond_bbox(arr, min_x, max_x, min_y, max_y)
            #arr = arr[condbbox]
            #inds = inds[condbbox]
            arr_ini = arr.copy()
        else:
            df_x = pd.read_csv(soilfile,delimiter=",",dtype = {'#index': int, 'z': float})
            
            arr = arr.copy()
            
            #print(df_x.values)
            if df_x.values.shape[0]>0:
                modinds, modz = df_x.values[:,0], df_x.values[:,1:2]              
                arr[np.isin(inds,modinds),2:3] = modz

        arr_sampled = arr[condbbox]

        x.append(arr_sampled)

    # Save wheel info
    wheelarr = np.transpose(np.array(wheelarr), (1,2,0))
    #print(wheelarr.shape)
    np.save(outpath+"/wheel",wheelarr.astype(np.float32))

    # Save vehicle info
    vehicl = np.transpose(np.array(vehicl), (1,0))
    #print(vehicl.shape)
    np.save(outpath+"/vehicle",vehicl.astype(np.float32))

    # Save soil info
    x = np.transpose(np.array(x), (1,2,0))
    #print(x.shape)
    np.save(outpath+"/soil",x.astype(np.float32))



# Routine to preprocess the full set of simulations
def preprocess_sim(pathsims, simname, pathoutput, lims=None):

    simname = simname.replace(pathsims, '')
    print("Processing",simname)

    #outpath = pathoutput+"/sim_{:d}".format(sim)
    outpath = pathoutput+simname

    if not os.path.exists(outpath):
        os.system("mkdir "+outpath)

    save_burst(pathsims+simname, outpath, maxtimesteps=None, lims=lims)

if __name__=="__main__":

    # Usage example: python PreprocessSCM.py --input=preNoNN/ --output=NoNN/
    parser = argparse.ArgumentParser()
    parser.add_argument('--simspath', 
                        help='simulation directory')
    parser.add_argument('--sim', 
                        help='sim name')
    parser.add_argument('--outpath', 
                        help='output directory')
    args = parser.parse_args()

    pathsims = args.simspath
    simname = args.sim
    pathoutput = args.outpath

    if not os.path.exists(pathoutput):
        os.system("mkdir "+pathoutput)

    # Frames per burst
    #maxtimesteps = None
    #maxtimesteps = 3500


    preprocess_sim(pathsims, simname, pathoutput, None)
