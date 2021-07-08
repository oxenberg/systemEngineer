import numpy as np
import matplotlib
import matplotlib.pyplot as plt


font = {'size'   : 22}

matplotlib.rc('font', **font)
plt.rcParams['legend.loc'] = "lower right"

labels = ['Last.FM', 'MovieLens', 'Book-Crossing']
McKR_AUC = [0.803, 0.913, 0.733]
MKR_AUC = [0.797, 0.917,  0.734]

McKR_ACC = [0.755, 0.839, 0.702]
MKR_ACC = [0.752, 0.843,  0.704]

McKR_time = [0.75, 2.41, 5.45]
MKR_time = [0.33, 1.03,  5.50]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

'''
Create the AUC graph
'''

fig, ax = plt.subplots(figsize=(10,10))
rects1 = ax.bar(x - width/2, McKR_AUC, width, label='McKR')
rects2 = ax.bar(x + width/2, MKR_AUC, width, label='MKR')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('AUC')
# ax.set_title('AUC')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

ax.bar_label(rects1, padding=6, label_type = 'edge')
ax.bar_label(rects2, padding=6 , label_type = 'edge')

# fig.tight_layout()

plt.show()

'''
Create the ACC graph
'''

fig, ax = plt.subplots(figsize=(10,10))
rects1 = ax.bar(x - width/2, McKR_ACC, width, label='McKR')
rects2 = ax.bar(x + width/2, MKR_ACC, width, label='MKR')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('ACC')
# ax.set_title('AUC')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

ax.bar_label(rects1, padding=6)
ax.bar_label(rects2, padding=6)

# fig.tight_layout()

plt.show()

'''
Create the time graph
'''

fig, ax = plt.subplots(figsize=(10,10))
rects1 = ax.bar(x - width/2, McKR_time, width, label='McKR')
rects2 = ax.bar(x + width/2, MKR_time, width, label='MKR')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Time[s]')
# ax.set_title('AUC')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

# fig.tight_layout()

plt.show()