import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegFileWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
# import cv2
from textwrap import wrap


def list_cut_average(ll, intervals):
    if intervals == 1:
        return ll

    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new = []
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new


def plot_3d_motion(save_path, kinematic_tree, joints, title, dataset, figsize=(3, 3), fps=120, radius=5,
                   vis_mode='default', gt_frames=[], view_params={'elev':120, 'azim':-90}, coloring_seq=None, extra_joints = None):
    matplotlib.use('Agg')

    title = '\n'.join(wrap(title, 20))

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        # print(title)
        fig.suptitle(title, fontsize=10)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    #         return ax
    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)
    print(data.min(), data[:, :, 2].min(), data[:, 11, 2].min(), joints[:, 11, 2].min())
    # preparation related to specific datasets
    if dataset == 'kit':
        data *= 0.003  # scale for visualization
    elif dataset == 'humanml':
        data *= 1.3  # scale for visualization
    elif dataset in ['humanact12', 'uestc']:
        data *= -1.5 # reverse axes, scale for visualization
    elif dataset in ['amass','amass_hml','h36m','3dpw']:
        data *= 1.
    # elif dataset in ['h36m','3dpw']:
    #     data *= -1. # reverse axes
    
    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = p3.Axes3D(fig)
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]  # GT color
    colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]  # Generation color
    colors = colors_orange
    if vis_mode == 'upper_body':  # lower body taken fixed to input motion
        colors[2] = colors_blue[2]
        colors[3] = colors_blue[3]
        colors[4] = colors_blue[4]
    elif vis_mode == 'gt':
        colors = colors_blue
    elif vis_mode == "left_arm":
        colors[4] = colors_blue[4]

    frame_number = data.shape[0]
    #     print(dataset.shape)
    data_cache = data
    ej_cache = extra_joints

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    frame0datax = data[:, 0:1, 0].copy()
    frame0dataz = data[:, 0:1, 2].copy()
    #data[..., 0] -= data[:, 0:1, 0]  # centering frame 0 in x
    #data[..., 2] -= data[:, 0:1, 2]  # centering frame 0 in z
    if extra_joints is not None:
        extra_joints = extra_joints.cpu().numpy()
        extra_joints[:,  1] -= height_offset
        extra_joints[:,0] -= frame0datax[:,0]
        extra_joints[:,2] -= frame0dataz[:,0]

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    #     print(trajec.shape)
    # import ipdb; ipdb.set_trace()
    def update(index):
        #         print(index)
        #plt.clf()
        #ax.lines = []
        #ax.collections = []
        ax.clear()
        ax.view_init(elev=view_params['elev'], azim=view_params['azim'])
        ax.dist = 7.5
        #         ax =
        #plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
        #             MAXS[2] - trajec[index, 1])

        plot_xzPlane(MINS[0] , MAXS[0] , 0, MINS[2] ,
                     MAXS[2] )
        #         ax.scatter(dataset[index, :22, 0], dataset[index, :22, 1], dataset[index, :22, 2], color='black', s=3)

        # if index > 1:
        #     ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
        #               trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
        #               color='blue')
        # #             ax = plot_xzPlane(ax, MINS[0], MAXS[0], 0, MINS[2], MAXS[2])

        used_colors = colors_blue if index in gt_frames else colors
        if coloring_seq is not None: 
            used_colors = colors_blue if coloring_seq[index] else colors_orange
        for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            if chain == 11:
                print(data[index, chain, 2])
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                      color=color)
        if extra_joints is not None:
            datapt = extra_joints[index]
            
            ax.scatter3D(datapt[0],datapt[1],datapt[2], color="pink")
        #         print(trajec[:index, 0].shape)
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        #plt.savefig("test.png")
    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)

    # writer = FFMpegFileWriter(fps=fps)
    ani.save(save_path, fps=fps)
    # ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False, init_func=init)
    # ani.save(save_path, writer='pillow', fps=1000 / fps)

    plt.close()
