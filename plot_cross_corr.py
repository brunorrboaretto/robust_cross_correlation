import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

sz1 = 12
sz2 = 10

window_size = 1200
delay = 200
data_list = [404,978,310]

for I in data_list:
    data01= np.loadtxt('cross_max_values_%.3d_ws_%.4d_delay_%.3d_p_%.2f.dat' % (I,window_size,delay,0.10))

    mua = np.loadtxt('data_MUA_%.3d.dat' % I)
    env = np.loadtxt('data_ENV_%.3d.dat' % I)

    fig = plt.figure()#constrained_layout=True)
    gs = fig.add_gridspec(4,1)

    ax_0 = fig.add_subplot(gs[0,0])
    ax_1 = fig.add_subplot(gs[1,0])
    ax_2 = fig.add_subplot(gs[2,0])
    ax_3 = fig.add_subplot(gs[3,0])

    ax_0.spines['top'].set_visible(False)
    ax_1.spines['top'].set_visible(False)
    ax_2.spines['top'].set_visible(False)
    ax_3.spines['top'].set_visible(False)

    ax_0.spines['right'].set_visible(False)
    ax_1.spines['right'].set_visible(False)
    ax_2.spines['right'].set_visible(False)
    ax_3.spines['right'].set_visible(False) 

    ax_0.tick_params(axis='both', labelsize=sz2)
    ax_1.tick_params(axis='both', labelsize=sz2)
    ax_2.tick_params(axis='both', labelsize=sz2)
    ax_3.tick_params(axis='both', labelsize=sz2)


    ax_0.set_ylabel(r'$X$ (MUA)',fontsize=sz1)
    ax_1.set_ylabel(r'$Y$ (ENV)',fontsize=sz1)
    ax_2.set_ylabel(r'$C^*$',fontsize=sz1)
    ax_3.set_ylabel(r'$\tau^*$',fontsize=sz1)
    ax_3.set_xlabel(r'time (ms)',fontsize=sz1)

    ax_0.set_xlim(0,len(env))
    ax_1.set_xlim(0,len(env))
    ax_2.set_xlim(0,len(env))
    
    ax_3.set_xlim(0,len(env))

    ax_3.text(len(env)+30,160,'(d)',fontsize=sz2)
    ax_3.text(len(env)+30,660,'(c)',fontsize=sz2)
    ax_3.text(len(env)+30,1160,'(b)',fontsize=sz2)
    ax_3.text(len(env)+30,1660,'(a)',fontsize=sz2)

    ax_2.set_ylim(-0.8,1.4)
    ax_3.set_ylim(-200,200)
    ax_2.grid()
    ax_3.grid()
    ax_0.set_xticks(np.arange(0, len(env), 1000))
    ax_1.set_xticks(np.arange(0, len(env), 1000))
    ax_2.set_xticks(np.arange(0, len(env), 1000))
    ax_3.set_xticks(np.arange(0, len(env), 1000))

    ax_2.set_yticks([-0.5,0,0.5,1])
    ax_3.set_yticks([-200,-100,0,100,200])

    ax_0.plot(mua,'k-',linewidth=.5)
    ax_1.plot(env,'k-',linewidth=.5)

    ax_2.fill_between(data01[:, 0], data01[:, 5] + data01[:, 6], data01[:, 7] - data01[:, 8], 
                  color='r', alpha=0.3,label='surr 1')  # Adjust alpha for transparency
    ax_2.fill_between(data01[:, 0], data01[:, 9] + data01[:, 10], data01[:, 11] - data01[:, 12], 
                  color='g', alpha=0.3,label='surr 2')  # Adjust alpha for transparency
    
    for i in range(len(data01) - 1):

        ax_2.plot(data01[i, 0], max(data01[i, 2],data01[i, 4]), 'mo',markersize=1, linewidth=2,label=r'$C^*$')#, label=r'surr $\tau_\mathrm{max} = 70$')
        if data01[i,-1] == 1:
            if data01[i, 2] > data01[i, 4]:
                ax_3.plot(data01[i, 0], data01[i, 1], 'ko',markersize=1, linewidth=2,label=r'$C^*$')#, label=r'surr $\tau_\mathrm{max} = 70$')
                ax_2.plot(data01[i, 0], data01[i, 2], 'ko',markersize=1, linewidth=2,label=r'$C^*$')#, label=r'surr $\tau_\mathrm{max} = 70$')
            else:
                ax_3.plot(data01[i, 0], data01[i, 3], 'ko',markersize=1, linewidth=2,label=r'$C^*$')#, label=r'surr $\tau_\mathrm{max} = 70$')
                ax_2.plot(data01[i, 0], data01[i, 4], 'ko',markersize=1, linewidth=2,label=r'$C^*$')#, label=r'surr $\tau_\mathrm{max} = 70$')
    frac = 0
    for i in range(len(data01)):
        MIN = data01[i, 7]
        frac+=MIN 
    frac/=len(data01)

    ax_0.set_title(r'Sample %.3d' % (I))

    legend_elements = [
    Line2D([0], [0], marker='o', color='k', markersize=5, label=r'$C^*$', linestyle=''),
    Line2D([0], [0], color='r', linewidth=2, label='Surrogate 1', linestyle='-'),  # Dashed line
    Line2D([0], [0], color='g', linewidth=2, label='Surrogate 2', linestyle='-')    # Dotted line
    ]

    ax_2.legend(handles=legend_elements,loc='upper right',fontsize=sz2-4,ncol=4)
    
    fp_out = 'plot_cross_%.3d.png' % (I)

    plt.subplots_adjust(left=None, bottom=.2, right=.6, top=None, wspace=.25, hspace=.3)
    width = 28; height = 16;
    fig.set_size_inches(width/2.54,height/2.54) #2.54 cm = 1 inches
    #plt.subplots_adjust(left=None, bottom=.2, right=None, top=None, wspace=None, hspace=None)
    plt.savefig(fp_out, dpi=300,bbox_inches='tight')