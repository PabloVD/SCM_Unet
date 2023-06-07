



#--------------------
"""
WARNING!

CHECK CLAMP VALUES FOR DEFORMATION, IT DEPENDS ON STEP SIZE!

"""
#-------------------

from Source.training import *
#from Source.network_torchscript import GNN
#from Source.network import *
#from Source.network_globnormglobframe import *
from Source.wrapper import *

from torch.profiler import profile, record_function, ProfilerActivity


def sampleinputdata(indbatch):

        condbatch = (batchsoil==indbatch)

        pos_soil_b = pos_soil[condbatch,:3]
        soil_vec = torch.cat([pos_soil_b,torch.zeros(pos_soil_b.shape[0],1)],dim=1) 

        wheeldata = [soil_vec, wheelpos[indbatch], orientquat[indbatch], glob[indbatch,:3], glob[indbatch,3:6]]

        return wheeldata

def wheeldir_t(quat):

        orient2d = torch.zeros(2, dtype=torch.float32)
        quat = quat.view(4,1)


        yax = torch.cat([ (quat[1] * quat[2] - quat[0] * quat[3]) * 2, (quat[0] * quat[0] + quat[2] * quat[2]) * 2 - 1, (quat[2] * quat[3] + quat[0] * quat[1]) * 2])
        zax = torch.tensor([0,0,1], dtype=torch.float32)

        xax = torch.cross(yax, zax)

        orient2d = xax[:2]/torch.norm(xax[:2])

        return orient2d


#----------------
# Create wrapper
#----------------

input_channels = 3
global_emb_dim = 6

model = Unet(input_channels = input_channels,
             num_layers = n_layers,
             hidden_channels_in = hidden_channels,
             global_emb_dim = global_emb_dim)

deltastep = 1
n_sims = 1
dataname = "DefSims"

namerun = "unet_"
namerun += dataname
namerun += "_test3"
namerun += "_deltastep_"+str(deltastep)
namerun += "_nsims_"+str(n_sims)


namerun += "_lays_{:d}_std_{:.1e}_chan_{:d}_batch_{:d}".format(n_layers, noise_std, hidden_channels, batch_size)
namerun += "_inputs_{:d}_globdim_{:d}".format(input_channels, global_emb_dim)

if use_wheeltype:
    namerun += "_wheeltype"

sufix = "_"+namerun+"_lrs_{:.1e}_{:.1e}".format(lr_min, lr_max)

bestmodelname = path+"models/bestmodel"+sufix
#bestmodelname = path+"models/bestrigmodel"+sufix
#bestmodelname = path+"models/lastmodel"+sufix
state_dict = torch.load(bestmodelname, map_location=device)
model.load_state_dict(state_dict)
print(bestmodelname)
#"""
model.to(device)

#exit()

model.eval()

wrapmodel = Wrapper(model, namerun)

script_wrapper = torch.jit.script(wrapmodel)

sufixx = "_"+str(device)
namewrapper = "wrapped_unet"+sufixx+".pt"
script_wrapper.save(namewrapper)
chronobuildpath = "/home/tda/CARLA/chrono_scm_newcode/build_def/data/vehicle/terrain/scm/"
#script_wrapper.save(chronobuildpath+namewrapper)


# Save in Unreal folder
#script_wrapper.save("/home/tda/CARLA/LastUnrealCARLA/"+"wrapped_gnn"+sufixx+".pt")

#----------------
# Prepare data
#----------------

n_sims = 1

maxtimesteps = 500

#pathchrono = "/home/tda/Descargas/SCM_simulations/OverfitSim"
pathchrono = "/home/tda/Descargas/SCM_simulations/FlatSettling/Valid"

train_dataset = load_chrono_dataset(pathsims=pathchrono, numsims=n_sims, maxtimesteps = maxtimesteps)

print("Sample graph:",train_dataset[0])


numwheels = 4

# Create data loaders

train_loader = DataLoader(train_dataset, batch_size=numwheels, shuffle=False, num_workers=12)


#step = -20
data = next(iter(train_loader))
#data.to(device)

time_tot = []



for step in range(maxtimesteps):

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        #----------------
        # Test wrapper in data
        #----------------


        pos = data.x[:,:,step]
        part_types = data.part_types
        condrig = (part_types==1)
        pos_soil = pos[~condrig]
        tirenodes = pos[condrig]
        glob = data.glob[:,:,step]
        wheeltype = data.wheeltype
        batch = data.batch
        batchsoil =  batch[~condrig]
        batchrig = batch[condrig]

        #hmap_nodes = torch.zeros((13041, 3))
        #hmap_nodes = torch.load("sink_hmap_nodes")

        #wheelpos = scatter(tirenodes, batchrig, dim=0, reduce="mean")
        wheelpos = data.wheelpos[:,:,step]
        #orientquat = torch.tensor([[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]],dtype=torch.float32)
        orientquat = data.quatorientation[:,:,step]

        # wheelframe = torch.zeros((numwheels,2))
        # for i in range(numwheels):
        #     wheelframe[i] = wheeldir_t(orientquat[i])

        #print(wheelpos.shape, pos.shape)


        #glob += 0.01*torch.randn(glob.shape)
        #glob[:,3] -= 1.
        #print("Vels",glob.mean(0))
        #print(data.force[:,:,step])

        

        #print(pos.shape, glob.shape, wheeltype.shape, wheelpos.shape)


        #"""
        w_0 = sampleinputdata(0)
        w_1 = sampleinputdata(1)
        w_2 = sampleinputdata(2)
        w_3 = sampleinputdata(3)


        #vehicle_info = torch.zeros(3,dtype=torch.float32)#

        loaded_wrapper = torch.jit.load(namewrapper)
        #loaded_wrapper = wrapmodel
        #loaded_wrapper.clampder = True
        #print("Clamp diff force:",loaded_wrapper.clampder)

        """
        hmap = loaded_wrapper.get_hmap(hmap_nodes)
        hmap = hmap.view(1,1,hmap.shape[0],hmap.shape[1])
        print("hmap",hmap_nodes.shape, hmap.shape)
        hmap = hmap.repeat(4,1,1,1)
        print("hmap",hmap_nodes.shape, hmap.shape)
        exit()
        """

        #prevstepsforce = data.force[:,:,step-prev_steps:step]
        #loaded_wrapper.prevstepsforce = prevstepsforce.clone()
        #print(torch.norm(loaded_wrapper.prevstepsforce[:,:3,-1],dim=1))
        #print("Ground truth Fz",torch.norm(data.force[:,:3,step],dim=1))
        #print("\nGround truth Fz",data.force[:,2,step])

        #print("\nLoaded wrapper")
        #print("Using overlap condition:", loaded_wrapper.overcondition)

        # For correctly clamping force
        """
        prevprev_force = data.force[:,:,step-2]
        prev_force = data.force[:,:,step-1]
        loaded_wrapper.prevprev_force = prevprev_force.clone()
        loaded_wrapper.prev_force = prev_force.clone()
        """
        #print("Prev force",loaded_wrapper.prev_force)


        out = loaded_wrapper(w_0, w_1, w_2, w_3, verbose=True)


        end.record()
        torch.cuda.synchronize()
        time_infer = start.elapsed_time(end)

        time_tot.append( time_infer )


prof.export_chrome_trace("trace.json")

burnphase = 50
time_tot = np.array(time_tot)
time_tot = time_tot[burnphase:]
bins = 100

plt.figure(figsize=(12,10))
plt.hist(time_tot, bins=bins )
#plt.plot(time_tot, linestyle=":", color="r" )
plt.title("Mean time: {:.1e} +- {:.1e} ms".format(time_tot.mean(), time_tot.std()))
#plt.yscale("log")
plt.xlabel("time [ms]")
plt.savefig("time.png")