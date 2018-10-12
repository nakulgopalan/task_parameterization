
import numpy as np
from sklearn.manifold import TSNE

from matplotlib import pyplot as plt


n = 100000

tsne = TSNE(n_components=2, verbose=2, perplexity=40, n_iter=1000)

results_mat = np.loadtxt('results100000.txt', dtype=float)[0:n,:]
tsne_results = tsne.fit_transform(results_mat[:,0:3])#np.loadtxt('results100000.txt', dtype=float)#

color_dict =['red','blue']
# plt.scatter(tsne_results[0:n,0], tsne_results[0:n,1], marker='o', alpha=.2, edgecolors=[color_dict[int(i)] for i in results_mat[0:n,-1] ] )
plt.scatter(tsne_results[0:n,0], tsne_results[0:n,1],  marker='o',  color=[color_dict[int(i)] for i in results_mat[0:n,-1] ] )

# alpha=.2,         ='red', markeredgewidth=5
# for label, x, y in zip(results_mat[0:n,-1],tsne_results[0:n,0],tsne_results[0:n,1]):
#     print(label)
#     if(label<0.5): color='red'
#     else: color='blue'
#     plt.scatter(x, y, c=color)

# np.savetxt('tsne_results.txt',tsne_results,fmt='%f')
#
plt.show()