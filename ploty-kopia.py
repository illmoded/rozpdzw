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





for i in range(0, 12):
  for j in range(i + 1, 12): 
    plt.scatter(X[:, i], X[:, j], c=Y, cmap=plt.cm.Paired)
	#plt.legend((d0,d1,d2,d3),('probka 1', 'probka 2', 'probka 3', 'probka 4'), loc='lower right')
    plt.xlabel(str(i))
    plt.ylabel(str(j))
    plt.savefig("pl/" + str(i) + "_" + str(j) + ".png")
    plt.close()
    pass
  pass

