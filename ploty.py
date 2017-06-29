# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 17:28:52 2017

@author: pawel
"""
import numpy as np
import matplotlib.pyplot as plt
from pepe import X,Y,clf,svc

"""
X - macierze
Y - osoby
clf - duże svc
svc - feed
"""

opis = ['justynka', 'madzia', 'pawel', 'krzysiu']
i = 4
j = 5
# for i in range(0, 12):
#   for j in range(i + 1, 12):
fig, ax = plt.subplots() 
ax.scatter(X[:, i], X[:, j], c=Y, cmap=plt.cm.Paired, label=Y)
ax.legend()
plt.xlabel(str(i))
plt.ylabel(str(j))
plt.savefig("plotz/" + str(i) + "_" + str(j) + ".png")
plt.show()

