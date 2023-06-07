import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import numpy as np
from IPython.display import HTML
import matplotlib as mpl

# Settings for plots
fontsiz = 20
mpl.rcParams.update({'font.size': fontsiz})
mpl.rcParams['xtick.labelsize'] = fontsiz
mpl.rcParams['ytick.labelsize'] = fontsiz
mpl.rcParams['xtick.major.size'] = fontsiz/2
mpl.rcParams['ytick.major.size'] = fontsiz/2
#mpl.rcParams['xtick.minor.size'] = fontsiz/4
#mpl.rcParams['ytick.minor.size'] = fontsiz/4

# this is for permutting axes from FleX datasets
permu = [0,2,1]

do_perm = False

indrig = 1


# Animation function
# It allows to plot either one or two axes
# See https://stackoverflow.com/questions/41602588/matplotlib-3d-scatter-animations
class new_graph_animation():
    def __init__(self, datax1, nindexes, time_ini=0, timesteps=100, datax2=None, interval=10, use_3d=False, show_error=True, forces=None, show_acc=False):

        self.time_ini = time_ini
        self.timesteps = timesteps
        self.interval = interval
        #self.dpi = 300
        self.dpi = 100
        self.fontsize = 20

        self.use_3d = use_3d
        self.show_error = show_error
        self.show_acc = show_acc

        self.datax1 = datax1
        self.datax2 = datax2

        #centroid = np.mean(self.datax1[:,:,0],axis=0)
        centroid = [0,0,0]

        #halfboxplot = 0.75
        halfboxplot = 1.

        self.limminx = centroid[0]-halfboxplot
        self.limminx2 = centroid[1]-halfboxplot#/2.
        self.limminx3 = centroid[2]-halfboxplot#/2.
        #self.limminx3 = -0.2
        #self.limmax = 3.2595882 #2.21
        self.limmax = centroid[0]+halfboxplot
        self.limmax2 = centroid[1]+halfboxplot#/2.
        self.limmax3 = centroid[2]+halfboxplot#/2.
        #self.limmax3 = 0.6 #2.

        """
        self.limminx = centroid[0]-0.5
        self.limminx2 = centroid[1]-0.5
        self.limminx3 = centroid[2]-0.5
        #self.limminx3 = -0.2
        #self.limmax = 3.2595882 #2.21
        self.limmax = centroid[0]+0.5
        self.limmax2 = centroid[1]+0.5
        #self.limmax3 = 0.6 #2.
        self.limmax3 = centroid[2]+0.5
        """

        self.forces = forces
        if self.forces is not None:
            self.show_force = True
            self.force_factor = 2.e-4#1.#e3
            self.centermass1 = forces[0]
            self.totalforce1 = forces[1]*self.force_factor
            if datax2 is not None:
                self.centermass2 = forces[2]
                self.totalforce2 = forces[3]*self.force_factor
            else:
                self.centermass2 = self.centermass1
                self.totalforce2 = self.totalforce1
        else:
            self.show_force = False


        #print(self.totalforce)
        #self.datax2 = np.zeros_like(datax1)
        self.quiv1 = None
        self.quiv2 = None
        self.quiv3 = None

        self.accstd = 5.e-2

        #self.colrig = "orange"
        #self.colfluid = "cornflowerblue"
        self.colrig = "grey"
        self.colfluid = "orange"

        self.nindexes = nindexes

        #self.fig = plt.figure(figsize=(20,8))
        self.fig = plt.figure(figsize=(20,8),frameon=False, dpi=self.dpi)
        #self.fig = plt.figure(figsize=(30,12),frameon=False, dpi=self.dpi)
        self.fig.tight_layout()
        self.fig.subplots_adjust(left=0, bottom=0, right=0.95, top=1, wspace=None, hspace=None)
        #if datasim2 is not None:  figsize=(12,5)
        #else:   figsize=(12,12)
        #fig = plt.figure(figsize=figsize)

        self.datax1 = self.datax1 - self.centermass1.reshape(1,3,-1)
        if datax2 is not None:
            self.datax2 = self.datax2 - self.centermass2.reshape(1,3,-1)

        self.centermass1 = np.zeros_like(self.centermass1)
        self.centermass2 = np.zeros_like(self.centermass2)

        #------------------
        # Truth subplot
        #------------------
        self.datafluid, self.datarigid = self.datax1[~nindexes], self.datax1[nindexes]
        self.ax1 = self.newaxis(1)
        self.scat_f = self.initaxis(self.ax1, self.datafluid, self.datarigid, "Truth")

        if self.show_force:
            self.quiv1 = self.plotforce(self.ax1, self.centermass1, self.totalforce1, 0)

        #------------------
        # Output subplot
        #------------------
        if datax2 is not None:

            self.datafluid2, self.datarigid2 = self.datax2[~nindexes], self.datax2[nindexes]
            self.ax2 = self.newaxis(2)
            self.scat2_f = self.initaxis(self.ax2, self.datafluid2, self.datarigid2, "Output")

            if self.show_force:
                self.quiv2 = self.plotforce(self.ax2, self.centermass2, self.totalforce2, 0)

        else:

            self.scat2_f = None

        #------------------
        # Error subplot
        #------------------
        if show_error and datax2 is not None:

            diff = np.mean(np.abs((datax2[:,:,time_ini]-datax1[:,:,time_ini])**2.),axis=1)
            #diff = diff[~self.nindexes]

            self.ax3 = self.newaxis(3)
            ax_3 = self.scat_e = self.initaxis(self.ax3, self.datafluid2, self.datarigid2, "Error=|Output-Truth|", diff=diff)
            cbar = plt.colorbar(ax_3, ax = self.ax3, fraction=0.046, pad=0.04)#, pad=0.15, anchor=(0.0, 0.5))

            if self.show_force:
                self.quiv3 = self.plotforce(self.ax3, self.centermass2, self.totalforce2-self.totalforce1, 0)

        else:

            self.scat_e = None

        #------------------
        # Acceleration subplot
        #------------------
        if show_acc and datax2 is not None:

            acc = np.abs(np.mean(self.datax2[:,:,self.time_ini+2] -2*self.datax2[:,:,self.time_ini+1] + self.datax2[:,:,self.time_ini],axis=1))/self.accstd

            #for ac in acc:
            #    print(ac)

            self.ax4 = self.newaxis(4)
            self.scat_a = self.initaxis(self.ax4, self.datafluid2, self.datarigid2, "Acceleration", diff=acc)

        else:

            self.scat_a = None

        #txt_title = fig.suptitle('Frame = 0')

    def animate(self):

        # blit=True re-draws only the parts that have changed.
        #anim = animation.FuncAnimation(fig, lambda n: self.drawframe(n, datafluid, datarigid, scat_f, scat_r, datafluid2, datarigid2, scat2_f, scat2_r, time_ini, use_3d), frames=timesteps, interval=interval, blit=True)
        anim = animation.FuncAnimation(self.fig, self.drawframe, frames=self.timesteps, interval=self.interval, blit=True)

        return anim

    def newaxis(self,numaxis):
        totaxes = 2
        if self.show_error:
            totaxes += 1
        if self.show_acc:
            totaxes += 1

        if self.use_3d:
            newax = self.fig.add_subplot(1,totaxes,numaxis, projection ="3d")
        else:
            newax = self.fig.add_subplot(1,totaxes,numaxis)

        return newax

    def initaxis(self, ax, datafluid, datarigid, title, diff=None):

        ax.set_xlim(self.limminx, self.limmax)
        ax.set_ylim(self.limminx2, self.limmax2)
        if self.use_3d:
            ax.set_zlim(self.limminx3, self.limmax3)

        # Fluid
        datasim = np.concatenate((datarigid, datafluid))
        pos = datasim[:,:,self.time_ini]


        colss = [self.colfluid if i == False else self.colrig for i in self.nindexes]

        if self.use_3d:
            if do_perm:
                pos = pos[:,permu]
            if diff is None:
                scat_f = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=colss, edgecolors='black')
            else:
                scat_f = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=diff, cmap='coolwarm', vmin=0., vmax=0.02)
        else:
            scat_f, = ax.plot(pos[:, 0], pos[:, 1], linestyle="", marker="o", c="blue")

        ax.set_title(title, fontsize=self.fontsize)

        return scat_f

    def update(self, datasim, scat, n, diff=None):

        pos = datasim[:,:,self.time_ini+n]

        if self.use_3d and do_perm: pos = pos[:,permu]

        scat._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])

        if diff is not None:
            scat.set_array(diff)

        return scat

    def plotforce(self, ax, centermass, totalforce, n):
        x, y, z = centermass[:,self.time_ini+n]
        fx, fy, fz = totalforce[:,self.time_ini+n]
        #print(x, y, z, fx, fy, fz)
        quiv = ax.quiver(x, y, z, fx, fy, fz, length=np.sqrt(fx**2. + fy**2. + fz**2.),color="red")
        return quiv

    # Update function, called sequentially
    #def drawframe(self, n, datafluid, datarigid, scat_f, scat_r, datafluid2=None, datarigid2=None, scat2_f=None, scat2_r=None, time_ini=0, use_3d=False):
    def drawframe(self, n):

        # Fluid
        datasim = np.concatenate((self.datarigid, self.datafluid))

        # Window
        #com_curr = self.datarigid.mean(0).view(1,3)
        #datasim = datasim - self.centermass1
        #window = sampleparts(relpos)

        self.scat_f = self.update(datasim, self.scat_f, n)

        if self.show_force:
            self.quiv1.remove()
            self.quiv1 = self.plotforce(self.ax1, self.centermass1, self.totalforce1, n)

        if self.datax2 is not None:

            # Fluid
            datasim2 = np.concatenate((self.datarigid2, self.datafluid2))
            #datasim2 = datasim2 - self.centermass2
            self.scat2_f = self.update(datasim2, self.scat2_f, n)

            if self.show_force:
                self.quiv2.remove()
                self.quiv2 = self.plotforce(self.ax2, self.centermass2, self.totalforce2, n)

        if self.show_error:

            #diff = np.mean(np.sqrt((self.datax2[:,:,self.time_ini+n]-self.datax1[:,:,self.time_ini+n])**2.),axis=1)
            diff = np.mean(np.abs(self.datax2[:,:,self.time_ini+n]-self.datax1[:,:,self.time_ini+n]),axis=1)
            diff = diff[~self.nindexes]

            #self.scat_e = self.update(self.datax2[~self.nindexes]- self.centermass2, self.scat_e, n, diff=diff)
            self.scat_e = self.update(self.datax2[~self.nindexes], self.scat_e, n, diff=diff)

            if self.show_force:
                self.quiv3.remove()
                self.quiv3 = self.plotforce(self.ax3, self.centermass2, self.totalforce2-self.totalforce1, n)

        if self.show_acc:

            acc = np.abs(np.mean(self.datax2[:,:,self.time_ini+2] -2*self.datax2[:,:,self.time_ini+1] + self.datax2[:,:,self.time_ini],axis=1))/self.accstd

            self.scat_a = self.update(self.datax2, self.scat_a, n, diff=acc)


        if self.datax2 is not None:

            if self.show_error:

                if self.show_force:
                    return self.scat_f, self.scat2_f, self.quiv1, self.quiv2, self.quiv3, self.scat_e,
                else:
                    #we = self.scat_f, self.scat2_f, self.scat_e,
                    #return we
                    return self.scat_f, self.scat2_f, self.scat_e,

            else:
                if self.show_force:
                    return self.scat_f, self.scat2_f, self.quiv1, self.quiv2, self.quiv3,
                else:
                    return self.scat_f, self.scat2_f,

        else:

            if self.show_force:
                return self.scat_f, self.quiv1,
            else:
                return self.scat_f,


def vis_results(data, gnn_out, in_step, lensteps, force_out, interval=10, seq=0):
    for btch in [0]:

        # Data is organized in batches. Take only the indexes from a single graph
        indexes = np.argwhere(data.batch.cpu()==btch).reshape(-1)

        gnn_outbtch = gnn_out.cpu().detach().numpy()[indexes]
        datasimbtch = data.x.cpu().detach().numpy()[indexes]



        part_types = data.part_types[indexes]
        nindexes = np.array([True if i == indrig else False for i in part_types], dtype=bool)
        #datafluid, datarigid = datasimbtch[~nindexes], datasimbtch[nindexes]
        #datafluid2, datarigid2 = gnn_outbtch[~nindexes], gnn_outbtch[nindexes]

        #cof1, cof2 = datasimbtch[nindexes].mean(0), gnn_outbtch[nindexes].mean(0)
        cof1, cof2 = data.wheelpos[btch].cpu().detach().numpy(), data.wheelpos[btch].cpu().detach().numpy()
        force1, force2 = data.force[btch].cpu().detach().numpy(), force_out[btch].cpu().detach().numpy()
        force1, force2 = force1[:3], force2[:3]
        forces = [cof1, force1, cof2, force2]

        anim = new_graph_animation(datasimbtch, nindexes, datax2=gnn_outbtch, forces=forces, time_ini=in_step, timesteps=lensteps, interval=interval, use_3d=True).animate()

        #anim = graph_animation(datasimbtch, time_ini=in_step, timesteps=lensteps, datasim2=gnn_outbtch, interval=20)
        display(HTML(anim.to_html5_video()))
        #"""
        dpi = 100
        writervideo = animation.writers['ffmpeg'](fps=10)
        anim.save("videos/test_"+str(seq)+".mp4",writer=writervideo,dpi=dpi)
        #"""
        plt.close()


def compare_truth_pred(data, gnn_out, step, writer):


    indexes = np.argwhere(data.batch.cpu()==0).reshape(-1)
    gnn_outbtch = gnn_out.cpu().detach().numpy()[indexes]
    datasimbtch = data.x.cpu().detach().numpy()[indexes]

    part_types = data.part_types[indexes]
    nindexes = np.array([True if i == indrig else False for i in part_types], dtype=bool)
    datafluid, datarigid = datasimbtch[~nindexes], datasimbtch[nindexes]
    datafluid2, datarigid2 = gnn_outbtch[~nindexes], gnn_outbtch[nindexes]

    #colormap = np.array(["orange","blue"])

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,6))
    centroid = datafluid[:,:,0].mean(0)
    ax1.set_xlim(centroid[0]-1.,centroid[0]+1.)
    ax1.set_ylim(centroid[1]-1.,centroid[1]+1.)
    ax2.set_xlim(centroid[0]-1.,centroid[0]+1.)
    ax2.set_ylim(centroid[1]-1.,centroid[1]+1.)
    """
    ax1.set_xlim(boundarylist[0][0],boundarylist[0][1])
    ax1.set_ylim(boundarylist[1][0],boundarylist[1][1])
    ax2.set_xlim(boundarylist[0][0],boundarylist[0][1])
    ax2.set_ylim(boundarylist[1][0],boundarylist[1][1])
    """
    ax1.scatter(datafluid[:,0,step],datafluid[:,1,step],c="blue")
    ax2.scatter(datafluid2[:,0,step],datafluid2[:,1,step],c="blue")
    ax1.scatter(datarigid[:,0,step],datarigid[:,1,step],c="orange")
    ax2.scatter(datarigid2[:,0,step],datarigid2[:,1,step],c="orange")
    writer.add_figure('Truth (left vs Prediction (right) at timestep '+str(step), fig)
    #plt.clf()


#writergif = animation.PillowWriter(fps=30)
#unused_animation.save("/home/pdomingo/CamelsGNN/TestGNNCarla/rolloutanimation.gif", writer=writergif)
