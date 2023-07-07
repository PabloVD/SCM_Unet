from Source.training import *
from Source.unet import Unet
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter


n_sims = None
#n_sims = 1

if dataname == "DefSims" or dataname == "DefSimsNoDamp":
    maxtimesteps = 2500
else:
    maxtimesteps = 500




firsttimestepvalid = 0
deltastep = 1

print("Memory being used (GB):",process.memory_info().rss/1.e9)



if use_rollout:
    train_dataset = load_chrono_dataset(pathsims=pathchrono, numsims=n_sims, maxtimesteps = maxtimesteps, use_rollout=True)
    if os.path.exists(pathchronoadd):
        train_dataset += load_chrono_dataset(pathsims=pathchronoadd, numsims=n_sims, maxtimesteps = maxtimesteps, use_rollout=True)
    valid_dataset = load_chrono_dataset(pathsims=pathvalid, numsims=n_sims, maxtimesteps = maxtimesteps, use_rollout=True)
    #train_dataset = create_rollout_dataset(train_dataset, rol_len=rol_len)
    #valid_dataset = create_rollout_dataset(valid_dataset, rol_len=rol_len)
    mxstps = valid_dataset[0].x.shape[2]

else:
    train_dataset = load_chrono_dataset(pathsims=pathchrono, numsims=n_sims, maxtimesteps = maxtimesteps, split_steps=True)
    if os.path.exists(pathchronoadd):
        train_dataset += load_chrono_dataset(pathsims=pathchronoadd, numsims=n_sims, maxtimesteps = maxtimesteps, split_steps=True)
    valid_dataset = load_chrono_dataset(pathsims=pathvalid, numsims=n_sims, maxtimesteps = maxtimesteps, split_steps=True)
    # random.shuffle(train_dataset)
    # valid_dataset = train_dataset[:int(0.1*len(train_dataset))]
    # train_dataset = train_dataset[int(0.1*len(train_dataset)):]

print(valid_dataset[0])

print(len(train_dataset), len(valid_dataset))

print("Data shape:",train_dataset[-1])

print("Memory being used (GB):",process.memory_info().rss/1.e9)

if deltastep > 1:
    train_dataset = sample_delta_dataset(train_dataset, deltastep)
    valid_dataset = sample_delta_dataset(valid_dataset, deltastep)

print("\nSample graph:",train_dataset[0])
#print(len(train_dataset),"sequences for training,",len(valid_dataset),"for testing")
#print("Data shape:",valid_dataset[-1].x.shape, valid_dataset[-1].glob.shape)

# Split training data in steps, such that each batch contains differents steps
#train_dataset = create_training_dataset(train_dataset)
print("\nSample graph:",train_dataset[0])

print("Memory being used (GB):",process.memory_info().rss/1.e9)



# Create data loaders

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)

input_channels = 3
global_emb_dim = 6

model = Unet(input_channels = input_channels,
             num_layers = n_layers,
             hidden_channels_in = hidden_channels,
             global_emb_dim = global_emb_dim)

model = model.to(device)

time_ini = time.time()

#model = GNN()

# Print the memory (in GB) being used now:
#process = psutil.Process()
#print("Memory being used (GB):",process.memory_info().rss/1.e9)

model.to(device)

#optimizer = torch.optim.Adam(model.parameters(), lr=lr_max, weight_decay=weight_decay)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr_max, weight_decay=weight_decay)



if sched_type == "CyclicLR":
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr_min, max_lr=lr_max, cycle_momentum=False)
elif sched_type =="ReduceLROnPlateau":
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=patience)
elif sched_type =="cos":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = patience, eta_min=lr_min)


namerun = "unet_"
namerun += dataname
namerun += "_test13"
namerun += "_deltastep_"+str(deltastep)
namerun += "_nsims_"+str(n_sims)
if use_log:
    namerun += "_log"
else:
    namerun += "_lin"
namerun += "_lrfact_{:.1e}".format(lr_fact)
namerun += "_gridsize_{:d}".format(sizegrid)
if use_rollout:
    namerun += "_rollout_"+str(rol_len)
else:
    namerun += "_singlestep"
namerun += "_margin_{:.1e}".format(margin)

namerun += "_lays_{:d}_std_{:.1e}_chan_{:d}_batch_{:d}".format(n_layers, noise_std, hidden_channels, batch_size)
namerun += "_inputs_{:d}_globdim_{:d}".format(input_channels, global_emb_dim)


if use_wheeltype:
    namerun += "_wheeltype"

sufix = "_"+namerun+"_lrs_{:.1e}_{:.1e}".format(lr_min, lr_max)
bestmodelname = path+"models/bestmodel"+sufix
lastmodelname = path+"models/lastmodel"+sufix
print("Model:", namerun+"_lrs_{:.1e}_{:.1e}".format(lr_min, lr_max))
if os.path.exists(lastmodelname):
    print("Loading previous model")
    print(lastmodelname)
    state_dict = torch.load(lastmodelname, map_location=device)
    model.load_state_dict(state_dict)
else:
    print("No previous model to be loaded")


tim = time.localtime()
strdate = "{:02d}-{:02d}_{:02d}:{:02d}".format(tim.tm_mday, tim.tm_mon, tim.tm_hour, tim.tm_min)
logdir = path+"runs/"+namerun+"_"+strdate



writer = SummaryWriter(log_dir=logdir)

print("\n")

#--- TRAINING LOOP ---#

train_losses, valid_losses = [], []
valid_loss_min = 1.e6
valid_loss_min_rig = 1.e6


train_step = 0
eval_per_epoch = 20

try:
    for epoch in range(1, num_epochs+1):

        print(f"Epoch: {epoch:02d}. Training")
        if use_rollout:
            train_loss, train_step = train_rollout(model, train_loader, optimizer, scheduler, train_step, writer, lastmodelname)
        else:
            train_loss, train_step = train(model, train_loader, optimizer, scheduler, train_step, writer, lastmodelname)
        
        train_losses.append(train_loss)
        writer.add_scalar("Training loss per epoch", train_loss, epoch)
        print(f'Epoch: {epoch:02d}, Training Loss: {train_loss:.2e}')
        #val_loss = test_singlesteps(model, valid_loader)
        #writer.add_scalar("Validation (single step) loss per epoch", val_loss, epoch)

        if (epoch+1)%eval_per_epoch==0:
            print(f"Epoch: {epoch:02d}. Validation")
            #print("Validation")
            #mxstps = min(110, maxtimesteps)

            if use_rollout:

                valid_loss = test(model, valid_loader, mxstps, writer)
                writer.add_scalar("Validation loss (rollout)", valid_loss, epoch)

            else:
            
                valid_loss = test_singlesteps(model, valid_loader)
                writer.add_scalar("Validation loss (single step)", valid_loss, epoch)
            
            #valid_loss, rig_loss = test(model, valid_loader, mxstps, writer)


            valid_losses.append(valid_loss)
            
            #writer.add_scalar("Rigid valid loss", rig_loss, epoch)
            print(f'Validation Loss: {valid_loss:.2e}')
            #print(f'Validation Rigid Loss: {rig_loss:.2e}')

            # Save model if it has improved
            if valid_loss <= valid_loss_min:

                #print("Epoch: {:02d}, Validation loss decreased ({:.2e} --> {:.2e}).  Saving model ...".format(epoch, valid_loss_min, valid_loss))
                valid_loss_min = valid_loss
                torch.save(model.state_dict(), bestmodelname)


        writer.flush()


except KeyboardInterrupt:
    print("\nInterrupted\n")
    pass




writer.close()



print("Finished. Time elapsed:",datetime.timedelta(seconds=time.time()-time_ini))



#state_dict = torch.load(bestmodelname, map_location=device)
#model.load_state_dict(state_dict)
#print(test(model, test_loader, maxtimesteps, writer, vis=False))
