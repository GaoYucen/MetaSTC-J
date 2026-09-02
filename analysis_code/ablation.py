# %% import the necessary packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pylab 
import matplotlib.patches as mpatches
from matplotlib.pyplot import MultipleLocator

# %% read the data from the csv file
data=pd.read_csv("data/ablation.csv")

# %% draw the bar plot of each application in a 8*2 subplot
figure=plt.figure(figsize=(6, 3))

# 修改字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

color=["#ADC865","#E3B168","#FF8357","#C93429", '#96B6C5']
# visit https://colorhunt.co/ to get the color

for i in range(5):
    # if i<4:
    #     plt.subplot(2,5,i+1)
    # else:
    #     plt.subplot(2,5,i+2)
    plt.subplot(1,5,i+1)
    plt.grid(axis="y",linestyle='--')
    plt.grid(visible=False, axis='x')
    plt.bar(data.iloc[0:5, 0].values,
            data.iloc[0:5, i+1].values,
            color=color,
            width=1)
    # plot a star above the smallest bar
    if i < 4:
        i_star=np.argmin(data.iloc[0:5, i+1].values)
        plt.plot(data.iloc[i_star, 0],
                 data.iloc[i_star, i+1]+5,
                 "*",
                 color=color[i_star],
                markersize=7)
    else:
        i_star=np.argmax(data.iloc[0:5, i+1].values)
        plt.plot(data.iloc[i_star, 0],
                 data.iloc[i_star, i+1]+0.05,
                 "*",
                 color=color[i_star],
                markersize=7)
    # if the value exceed, then put the tag of the value on the bar.
    # y_major_locator=MultipleLocator(0.1)
    ax=plt.gca()
    # ax.yaxis.set_major_locator(y_major_locator)
    if i < 4:
        plt.ylim(0,100)
    else:
        plt.ylim(0,1)
        # put y lim on the right
        # ax.yaxis.tick_right()
        # ax.yaxis.set_label_position("right")
    pylab.setp(ax.get_xticklabels(), visible=False)
    
    if i in [1,2,3]:
        pylab.setp(ax.get_yticklabels(), visible=False)
    elif i == 0:
        ax.yaxis.tick_left()
    else:
        ax.yaxis.tick_right()

    # print(data.columns[i+1])
    plt.title(data.columns[i+1],y=-0.12)
# plt.subplot(1,5,5)
# plt.grid(axis="y",linestyle='--')
# plt.grid(visible=False, axis='x')
# plt.bar(data.iloc[0:4, 0].values,
#         data.iloc[0:4, 9].values,
#         color=color,
#         width=1)
# i_star=np.argmin(data.iloc[0:4, 9].values)
# plt.plot(data.iloc[i_star, 0],
#         data.iloc[i_star, 9]+0.01,
#         "*",
#         color=color[i_star],
#         markersize=15,
#         )
# for j in range(4):
#     if data.iloc[j, 9]>0.2:
#         plt.text(data.iloc[j, 0],
#                     0.28,
#                     str("0.8"),
#                     ha='center',
#                     va='bottom',
#                     fontsize=8)
# ax=plt.gca()
# plt.ylim(0,0.3)
# # put y lim on the right
# ax.yaxis.tick_right()
# ax.yaxis.set_label_position("right")
# pylab.setp(ax.get_xticklabels(), visible=False)
# plt.title(data.columns[9],y=-0.12)
# plt.ylabel("STCMN")

# draw the legend
legend_elements = []
for i in range(4):
    legend_elements.append(mpatches.Patch(facecolor=color[i], label=data.iloc[i, 0]))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.25)

figure.legend(handles=legend_elements, loc='upper center',ncol=5, bbox_to_anchor=(0.5, 1.03),fancybox=True, shadow=True,fontsize=9)

# %% save the figure to MCES.pdf
# plt.show()
figure.savefig("graph/ablation.pdf", bbox_inches='tight')
