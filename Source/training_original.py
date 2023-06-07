
#from Source.rigid_body import *
from Source.utils import *
from Source.visualize import *

rig_soil_fact = 1. #1.
sample_loss = False

if use_amp:
    # Automated Mixed Precision
    scaler = torch.cuda.amp.GradScaler()

#--- TRAINING ---#

if use_cosineloss:
    criterion_r = torch.nn.L1Loss(reduction='sum')
    #criterion_r = torch.nn.SmoothL1Loss(reduction='sum')
else:
    criterion_r = torch.nn.MSELoss(reduction='sum')
criterion_s = torch.nn.MSELoss(reduction='sum')
#cosineemb = torch.nn.CosineEmbeddingLoss(reduction='sum')
#criterion = lambda x, y: cosineemb(x, y, 1)
cosinesim = torch.nn.CosineSimilarity()
#criterion = lambda x, y: torch.sum(1. - cosinesim(x, y),dim=0)



# Predict the following state at timestep step and compute loss
def singlestep(model, data, step, use_noise=True):

    x, btch, part_types = data.x, data.batch, data.part_types

    glob = data.glob[:,:,step]
    #glob = torch.cat([data.glob[:,:,step], data.wheeltype], dim=1)
    forcestep = data.force[:,:,step]
    #forcestep = data.force[:,:,step+1]  # Jan18

    condrig = (part_types==1)
    batchrig = btch[condrig]
    batchsoil = btch[~condrig]

    # SCM
    dataseq = x[:,:,step]

    # Inject noise
    if use_noise:
        dataseq += noise_std*torch.randn(size=dataseq.shape,device=device)

    if use_sinkage:
        #sink = x[:,2:3,0] - x[:,2:3,step]
        sink = x[:,2:3,0] - dataseq[:,2:3]
        sink[condrig] = 0.
        dataseq = torch.cat([dataseq, sink],dim=1)

    if use_vel:
        # allvel
        vel = tire_vel(global_mean_pool(dataseq[condrig,:3], batchrig), dataseq[:,:3], glob[:,:3], glob[:,3:6], btch)/vel_std
        #vel = torch.zeros((dataseq.shape[0], 3), device=device)
        #vel[condrig] = tire_vel(global_mean_pool(dataseq[condrig,:3], batchrig), dataseq[condrig,:3], glob[:,:3], glob[:,3:6], batchrig)/vel_std
        #vel[~condrig] = (dataseq[~condrig,:3] - x[~condrig,:3,step-1])/dt/vel_std
        dataseq = torch.cat([dataseq, vel],dim=1)

    # sinkageprev
    #"""
    sinkprev = x[:,2:3,0] - x[:,2:3,step-1]
    sinkprev[condrig] = 0.
    dataseq = torch.cat([dataseq, sinkprev],dim=1)
    #"""

    prevstep, currstep, nextstep = x[:,:,step-1], dataseq[:,:3], x[:,:,step+1]

    currst = currstep #+ noise_seq[:,:,-1]
    prevst = prevstep #+ noise_seq[:,:,-2]
    nextst = nextstep #+ noise_seq[:,:,-1]
    currrig, prevrig, nextrig = currst[condrig], prevst[condrig], nextst[condrig]
    currsoil, prevsoil, nextsoil = currst[~condrig], prevst[~condrig], nextst[~condrig]

    # Window
    com_curr = global_mean_pool(currrig, batchrig)
    relpos = dataseq[:,:3] - com_curr[btch]
    window = sampleparts(relpos)
    dataseq_w = dataseq[window]
    part_types_w = part_types[window]
    btch_w = btch[window]

    # Random drop of particles
    #"""
    if use_noise and droprate>0:
        windowdrop = torch.cuda.FloatTensor(dataseq_w.shape[0]).uniform_() > droprate
        dataseq_w = dataseq_w[windowdrop]
        part_types_w = part_types_w[windowdrop]
        btch_w = btch_w[windowdrop]
    #"""
    
    """
    pos_seq = x[:,:,step-seq_len:step+1]  # index 0 is initial step, -1 is last instant, current step
    noise_seq = get_noise(pos_seq, noise_std).to(device)
    pos_seq += noise_seq
    dataseq = data_prev_steps(pos_seq)

    dataseq[condrig,3:] = tire_vel(global_mean_pool(dataseq[condrig,:3], batchrig), dataseq[condrig,:3], glob[:,:3], glob[:,5:8], batchrig)/vel_std + noise_seq[condrig,:,-1]/dt/vel_std
    """

    outsoil, outrig = model(dataseq_w, btch_w, part_types_w, glob, data.wheeltype, data.soiltype, data.wheelframe[:,:,step])   # predict acceleration (or displacement)

    # Soil loss

    # SCM
    """
    acc_soil_tar = get_acceleration(prevsoil, currsoil, nextsoil)

    # adjust for noise with last value, leaving velocity at current step unchanged for consistency
    acc_soil_tar = acc_soil_tar - noise_seq[~condrig,:,-1]/dt**2./acc_std



    if sample_loss:
        com_curr = global_mean_pool(currrig, batchrig)
        distparticles = torch.norm(currsoil - com_curr[data.batch[~condrig]],dim=1)
        closeparticles = (distparticles - 2.*wheel_radius <= 0.)

        loss_soil = criterion(outsoil[closeparticles], acc_soil_tar[closeparticles])#/nextsoil.shape[0]

    else:
        loss_soil = criterion(outsoil, acc_soil_tar)#/acc_soil_tar.shape[0]
    """
    if use_noise and droprate>0:
        curr_step, next_step = dataseq[:,:3], x[:,:,step+1]
        def_true = next_step[:,2:3] - curr_step[:,2:3]
        def_true = def_true[window]
        def_true = def_true[windowdrop]
        def_true = def_true[part_types_w==0]
    else:
        def_true = nextsoil[:,2:3]-currsoil[:,2:3]
        def_true = def_true[window[~condrig]]
    loss_soil = criterion_s(outsoil, def_true)

    # Rigid loss

    #acc_com, quat = outrig[:,:3], outrig[:,3:]
    #acc_com, torque = outrig[:,:3], outrig[:,3:]
    #print(acc_com.shape, torque.shape)
    #vel = (currrig-prevrig)/vel_std

    normforcestep = norm_force(forcestep[:,:3])
    if not use_cosineloss:
        loss_rig = criterion_r(normforcestep, outrig[:,:3])
        logloss_rig = torch.log10(loss_rig)
    else:
        # cosineloss
        #"""
        modloss = criterion_r(torch.norm(normforcestep,dim=1), torch.norm(outrig[:,:3],dim=1))
        cosloss = torch.sum(1. - cosinesim(normforcestep, outrig[:,:3]),dim=0)
        logloss_rig = torch.log10(modloss) + torch.log10(cosloss)
        #"""
    

    if use_torque:
        normtorqstep = norm_torque(forcestep[:,3:])
        if not use_cosineloss:
            loss_rig_torq = criterion_r(normtorqstep, outrig[:,3:])
            logloss_rig += torch.log10(loss_rig_torq)
        else:
            # cosineloss
            #"""
            modloss = criterion_r(torch.norm(normtorqstep,dim=1), torch.norm(outrig[:,3:],dim=1))
            cosloss = torch.sum(1. - cosinesim(normtorqstep, outrig[:,3:]),dim=0)
            logloss_rig += torch.log10(modloss) + torch.log10(cosloss)
            #"""

    """
    rigloss_x = criterion(outrig[:,0], normforcestep[:,0])
    rigloss_y = criterion(outrig[:,1], normforcestep[:,1])
    rigloss_z = criterion(outrig[:,2], normforcestep[:,2])
    logloss_rig = torch.log10(rigloss_x) + torch.log10(rigloss_y) + torch.log10(rigloss_z)
    """

    """

    com_curr = global_mean_pool(currrig, batchrig)
    com_prev = global_mean_pool(prevrig, batchrig)

    pos_relcom = currrig-com_curr[batchrig]

    if use_torque:
        acc_com, torque = outrig[:,:3], outrig[:,3:]

        acc_rig_tar = get_acceleration(prevrig, currrig, nextrig)
        acc_com_tar = global_mean_pool(acc_rig_tar, batchrig)
        torque_tar = get_torque(pos_relcom, acc_rig_tar, batchrig)

        loss_rig1 = criterion(acc_com, acc_com_tar)
        loss_rig2 = criterion(torque, torque_tar)
        logloss_rig = torch.log10(loss_rig1) + torch.log10(loss_rig2)

    else:
        acc_com, quat = outrig[:,:3], outrig[:,3:]

        com_next = pred_position(acc_com, com_curr, com_prev)

        RotM = rotation_matrix_from_quaternion(quat, device)
        posrig = pos_rig_bodynew(pos_relcom, com_next[batchrig], RotM[batchrig])
        #print(posrig.shape, com_next.shape, RotM.shape)
        loss_rig = criterion(posrig, nextrig)#/posrig.shape[0]+
        logloss_rig = torch.log10(loss_rig)

    """

    #print(torch.log10(loss_soil).item(), logloss_rig.item())

    loss = torch.log10(loss_soil) + rig_soil_fact*logloss_rig

    #pred_pos_soil = pred_position(outsoil, currsoil, prevsoil)
    #loss_soilpos = criterion(pred_pos_soil, nextsoil)
    #print(loss_soilpos, loss_rig)

    return loss

# Predict sequence (rollout) starting at timestep in_step with a length rollout_seq and compute loss
def rollout(model, data, in_step=seq_len, rollout_seq=50):

    part_types = data.part_types

    loss_rol = 0
    loss_rol_rig = 0

    maxtimesteps = data.x.shape[2]

    btch = data.batch

    if in_step+rollout_seq>maxtimesteps-1:
        print("max step too large",in_step+rollout_seq-1)

    # Initialize the prediction tensor
    gnn_out = torch.zeros_like(data.x, device=device)
    #glob_out = torch.zeros_like(data.glob, device=device)
    #glob_out = data.glob
    force_out = torch.zeros_like(data.force, device=device)

    #"""
    for step in range(in_step-seq_len,in_step+1):
        gnn_out[:,:,step] = data.x[:,:,step]
        force_out[:,:,step] = data.force[:,:,step]
    #"""

    condrig = (part_types==1)
    batchrig = btch[condrig]

    # SCM
    gnn_out = data.x.clone()    # Afterwards, this is updated!
    #gnn_out[:,:2] = data.x[:,:2]
    #gnn_out[condrig] = data.x[condrig]

    # For each time "step" and previous ones, predict step+1
    steps = range(in_step, in_step+rollout_seq)
    for step in steps:

        glob = data.glob[:,:,step]
        #glob = torch.cat([data.glob[:,:,step], data.wheeltype], dim=1)

        #loss += singlestep(data.x, data.batch, step)
        dataseq = gnn_out[:,:,step]

        if use_sinkage:
            sink = gnn_out[:,2:3,0] - gnn_out[:,2:3,step]
            sink[condrig] = 0.
            dataseq = torch.cat([dataseq, sink],dim=1)

        if use_vel:
            
            # allvel
            vel = tire_vel(global_mean_pool(dataseq[condrig,:3], batchrig), dataseq[:,:3], glob[:,:3], glob[:,3:6], btch)/vel_std
            #vel = torch.zeros((dataseq.shape[0], 3), device=device)
            #vel[condrig] = tire_vel(global_mean_pool(dataseq[condrig,:3], batchrig), dataseq[condrig,:3], glob[:,:3], glob[:,3:6], batchrig)/vel_std
            #vel[~condrig] = (dataseq[~condrig,:3] - gnn_out[~condrig,:3,step-1])/dt/vel_std
            dataseq = torch.cat([dataseq, vel],dim=1)

        # sinkageprev
        #"""
        sinkprev = gnn_out[:,2:3,0] - gnn_out[:,2:3,step-1]
        sinkprev[condrig] = 0.
        dataseq = torch.cat([dataseq, sinkprev],dim=1)
        #"""

        nextstep = data.x[:,:,step+1]#.clone()
        prevstep, currstep = gnn_out[:,:,step-1], gnn_out[:,:,step]

        currst = currstep
        prevst = prevstep
        nextst = nextstep
        currrig, prevrig, nextrig = currst[condrig], prevst[condrig], nextst[condrig]
        currsoil, prevsoil, nextsoil = currst[~condrig], prevst[~condrig], nextst[~condrig]

        # Window
        com_curr = global_mean_pool(currrig, batchrig)
        relpos = dataseq[:,:3] - com_curr[btch]
        window = sampleparts(relpos)
        dataseq_w = dataseq[window]
        part_types_w = part_types[window]
        btch_w = btch[window]

        outsoil_w, outrig = model(dataseq_w, btch_w, part_types_w, glob, data.wheeltype, data.soiltype, data.wheelframe[:,:,step])
        #print(dataseq[~condrig].shape, outsoil_w.shape, z_pred.shape)

        # Soil
        outsoil = torch.zeros((dataseq[~condrig].shape[0],1), device=device)
        outsoil[window[~condrig]] = outsoil_w

        # SCM
        """
        pred_pos_soil = pred_position(outsoil, currsoil, prevsoil)
        #loss_rol += criterion(pred_pos_soil, nextsoil)

        if sample_loss:

            com_curr = global_mean_pool(currrig, batchrig)
            distparticles = torch.norm(currsoil - com_curr[data.batch[~condrig]],dim=1)
            closeparticles = (distparticles - 2.*wheel_radius <= 0.)

            soilloss = criterion(pred_pos_soil[closeparticles], nextsoil[closeparticles])#/nextsoil.shape[0]ç

        else:
            soilloss = criterion(pred_pos_soil, nextsoil)#/nextsoil.shape[0]

        """
        #def_true = nextsoil[:,2:3]-currsoil[:,2:3]
        z_pred = currsoil[:,2:3] + outsoil
        z_pred[~window[~condrig]] = nextsoil[~window[~condrig],2:3]

        #soilloss = criterion(outsoil, def_true)
        soilloss = criterion_s(z_pred[window[~condrig]], nextsoil[window[~condrig],2:3])
        #soilloss = torch.abs(outsoil - def_true).mean()

        #print(z_pred.shape, outsoil.shape, dataseq.shape, gnn_out.shape, window.shape, window[~condrig].shape)

        # Rigid
        """
        acc_com, rigrotdof = outrig[:,:3], outrig[:,3:]

        com_curr = global_mean_pool(currrig, batchrig)
        com_prev = global_mean_pool(prevrig, batchrig)

        pos_relcom = currrig-com_curr[batchrig]

        com_next = pred_position(acc_com, com_curr, com_prev)

        if use_torque:
            vel = (currrig-prevrig)/vel_std
            RotM = get_rotation(rigrotdof, pos_relcom, vel, batchrig)
        else:
            RotM = rotation_matrix_from_quaternion(rigrotdof, device)
        pred_pos_rig = pos_rig_bodynew(pos_relcom, com_next[batchrig], RotM[batchrig])

        loss_rol += criterion(pred_pos_rig, nextrig)
        """
        
        """
        rigloss_x = criterion(outrig[:,0], normforcestep[:,0])
        rigloss_y = criterion(outrig[:,1], normforcestep[:,1])
        rigloss_z = criterion(outrig[:,2], normforcestep[:,2])
        logloss_rig = torch.log10(rigloss_x) + torch.log10(rigloss_y) + torch.log10(rigloss_z)
        """
        #print(outrig[0,2], normforcestep[0,2])

        forcestep = data.force[:,:,step]
        #forcestep = data.force[:,:,step+1]  # Jan18
        normforcestep = norm_force(forcestep[:,:3])
        #normforcestep = forcestep
        rigloss = criterion_r(normforcestep, outrig[:,:3])
        logloss_rig = torch.log10(rigloss)
        #print(outrig[:,3:])

        if use_torque:
            normtorqstep = norm_torque(forcestep[:,3:])
            loss_rig_torq = criterion_r(normtorqstep, outrig[:,3:])
            logloss_rig += torch.log10(loss_rig_torq)

        #print(torch.log10(soilloss).item(), torch.log10(rigloss).item())

        #loss_rol += torch.log10(soilloss) + logloss_rig
        loss_rol += soilloss
        #loss_rol_rig += logloss_rig

        # Rigid absolute error
        unnormforce = unnorm_force(outrig[:,:3])
        rigloss = torch.abs(unnormforce-data.force[:,:3,step]).mean()
        loss_rol_rig += rigloss

        # SCM
        #gnn_out[~condrig,2:3,step+1] = z_pred

        #print(gnn_out[window,2:3,step+1].shape)
        #print(gnn_out[window,2:3,step+1][~condrig].shape)
        #print(gnn_out[window[~condrig],2:3,step+1].shape)


        #gnn_out[window,2:3,step+1][~condrig] = z_pred[window[~condrig]]
        gnn_out[~condrig,2:3,step+1] = z_pred
        

        #velrig = pred_pos_rig - currrig
        #glob_out[:,:2,step+1] = get_orientation(velrig, glob_out[:,:2,step], batchrig)

        #print("Truth:", forcestep[:,:3].view(-1).tolist(), forcestep[:,3:].view(-1).tolist(), "Out:", unnorm_force(outrig[:,:3]).view(-1).tolist(), unnorm_torque(outrig[:,3:]).view(-1).tolist())
        #print("Truth:", forcestep[:,:3].view(-1).tolist(), "Out:", unnorm_force(outrig[:,:3]).view(-1).tolist())


        #force_out[:,:,step] = outrig
        #"""
        force_out[:,:3,step] = unnorm_force(outrig[:,:3])
        if use_torque:
            force_out[:,3:,step] = unnorm_torque(outrig[:,3:])
        #"""
        # Jan18
        """
        force_out[:,:3,step+1] = unnorm_force(outrig[:,:3])
        if use_torque:
            force_out[:,3:,step+1] = unnorm_torque(outrig[:,3:])
        """

    #print(force_out[0,:,-2])


    return loss_rol/len(steps), loss_rol_rig/len(steps), gnn_out, force_out



# Training routine
def train(model, loader, optimizer, scheduler, train_step, writer, lastmodelname):
    model.train()

    optimizer.zero_grad(set_to_none=True)

    total_loss = 0
    pbar = tqdm(loader, total=len(loader), position=0, leave=True, desc=f"Training")
    for data in pbar:

        data = data.to(device)

        # Train with single steps
        # For each time "step" and previous ones, predict step+1
        maxtimesteps = data.x.shape[2]
        #numsteps = min(80, maxtimesteps-1-seq_len)
        #steps = random.choices(range(seq_len, maxtimesteps-1),k=numsteps)
        steps = list(range(0, maxtimesteps-1))
        random.shuffle(steps)

        for i, step in enumerate(steps):

            if use_amp:
                with torch.cuda.amp.autocast():
                    loss = singlestep(model, data, step)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = singlestep(model, data, step)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()


            if scheduler is not None:
                scheduler.step()

            loss = loss.item()/len(steps)
            total_loss += loss
            if train_step % log_steps == 0:
                writer.add_scalar("Training loss per gradient step", loss, train_step)

            optimizer.zero_grad(set_to_none=True)  # Clear gradients.
            train_step+=1

            if (train_step+1)%steps_save_model==0:
                torch.save(model.state_dict(), lastmodelname)

    return total_loss / len(loader), train_step



# Evaluation routine
@torch.no_grad()
def test(model, loader, maxsteps, writer, vis=False):
    model.eval()

    total_loss = 0.
    tot_loss_rig = 0.
    for data in loader:
        data = data.to(device)
        loss = 0

        loss_rol, loss_rig, gnn_out, force_out = rollout(model, data, in_step=seq_len, rollout_seq=maxsteps-seq_len-1)
        #print(gnn_out[:3,:,-3:])
        #print(force_out[:,:,-3:])
        total_loss += loss_rol.item()#/data.x.shape[0]
        tot_loss_rig += loss_rig.item()#/data.x.shape[0]
        if vis:
            vis_results(data, gnn_out, in_step=0, lensteps=maxsteps-1, force_out=force_out, interval=100)

    #if writer is not None:
    #    compare_truth_pred(data, gnn_out, -1, writer, boundarylist)
        #compare_truth_pred(data, gnn_out, 60, writer, boundarylist)

    return total_loss / len(loader), tot_loss_rig / len(loader)

@torch.no_grad()
def test_singlesteps(model, loader):
    model.eval()

    total_loss = 0
    pbar = tqdm(loader, total=len(loader), position=0, leave=True, desc=f"Validation single steps")
    for data in pbar:

        data = data.to(device)

        # Train with single steps
        maxtimesteps = data.x.shape[2]
        steps = list(range(0, maxtimesteps-1))
        random.shuffle(steps)

        for i, step in enumerate(steps):

            loss = singlestep(model, data, step, use_noise=False)

            loss = loss.item()/len(steps)
            total_loss += loss
            #if train_step % log_steps == 0:
            #    writer.add_scalar("Training loss per gradient step", loss, train_step)

    return total_loss / len(loader)


box_search_depth = 0.2
box_search_lateral = 0.2
box_search_forward = 0.5

def subsample_func(soilpos, soilvel, wheelpos):

    inds = (soilpos[:,2] > wheelpos[2]-wheel_radius-box_search_depth)
    soilpos, soilvel = soilpos[inds], soilvel[inds]

    inds = (soilpos[:,0] > wheelpos[0]-box_search_forward)
    soilpos, soilvel = soilpos[inds], soilvel[inds]

    inds = (soilpos[:,0] < wheelpos[0]+box_search_forward)
    soilpos, soilvel = soilpos[inds], soilvel[inds]

    inds = (soilpos[:,1] > wheelpos[1]-box_search_lateral)
    soilpos, soilvel = soilpos[inds], soilvel[inds]

    inds = (soilpos[:,1] < wheelpos[1]+box_search_lateral)
    soilpos, soilvel = soilpos[inds], soilvel[inds]

    return soilpos, soilvel



# relsoil must be rotated wrt wheel
def sampleparts(relpos):

    rad_hor = torch.sqrt(relpos[:,0]**2. + relpos[:,1]**2.)
    cond_hor = (rad_hor < window_radius)
    cond_vert = (relpos[:,0] < 0.)
    condition = torch.logical_and(cond_hor, cond_vert)
    #condition = cond_hor
    
    return condition