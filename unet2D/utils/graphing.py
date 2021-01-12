import numpy as np
import pandas
import matplotlib.pyplot as plt
import seaborn as sns

def ct_pt_and_boxplot(fig, ax, violin_data, swarm_data, stat_header, orientation):
    # points subplot
    if orientation == 'h':
        ax.set_title(stat_header, y=1.0, pad=-14)
        ax.tick_params(direction = 'in')
        sns.violinplot(fig = fig,
                    ax = ax,
                    data = violin_data[stat_header],
                    orient = 'h',
                    width = 0.5,
                    legend = False,
                    scale = 'area',
                    inner = None, 
                    cut = 0,
                    bw = 0.5)
        plt.setp(ax.collections, alpha = 0.1)
        plt.scatter(swarm_data[stat_header][0], 0, 
                    marker='o', 
                    s=60, 
                    c = 'blue')
        ax.annotate(swarm_data[stat_header][0], 
                    (swarm_data[stat_header][0], 0), 
                    textcoords = 'offset pixels', xytext = (0, 7), 
                    fontsize = 12, 
                    horizontalalignment='center')
        sns.swarmplot(swarm_data[stat_header][1:], 
                      np.zeros(len(swarm_data[stat_header]) - 1), 
                      ax = ax, orient = 'h', 
                      color = 'blue', 
                      alpha = 0.2, 
                      size = 6)
        ax.get_yaxis().set_visible(False)
        ax.set_xlim(left = 0)
        ax.set_ylim(bottom = -0.35, top = 0.45)
        ax.axes.get_xaxis().get_label().set_visible(False)
    else:
        ax.set_title(stat_header, y=1.0, pad=-14)
        ax.tick_params(direction = 'in')
        sns.violinplot(fig = fig,
                       ax = ax,
                       data = violin_data[stat_header],
                       orient = 'v',
                       width = 0.5,
                       legend = False,
                       scale = 'area',
                       inner = None, 
                       cut = 0,
                       bw = 0.5)
        plt.setp(ax.collections, alpha = 0.1)
        plt.scatter(0, swarm_data[stat_header][0],
                    marker='o', 
                    s=60, 
                    c = 'blue')
        ax.annotate(swarm_data[stat_header][0], 
                    (0, swarm_data[stat_header][0]), 
                    textcoords = 'offset pixels', xytext = (7, 7), 
                    fontsize = 12, 
                    horizontalalignment='center')
        sns.swarmplot(np.zeros(len(swarm_data[stat_header]) - 1), 
                      swarm_data[stat_header][1:], 
                      ax = ax, orient = 'v', 
                      color = 'blue', 
                      alpha = 0.2, 
                      size = 6)
        ax.get_xaxis().set_visible(False)
        ax.set_ylim(bottom = 0, top = violin_data[stat_header].max()*1.15)
        ax.axes.get_yaxis().get_label().set_visible(False)