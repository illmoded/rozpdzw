# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 00:39:38 2017

@author: pawel
"""

from voice import voice
from peppa import dataset
from sklearn import svm
import sklearn.datasets.base
import numpy as np

import warnings
warnings.filterwarnings("ignore")

def czytaj_plik(nazwa):
  sygnal = voice(nazwa)
  czemu_pawel = sygnal.mfcc
  L, D = czemu_pawel[1].shape
  LD = L*D
  A = np.empty([0,LD])
  name = []

  for smffc in czemu_pawel:
      a = np.array(smffc).ravel().reshape(1, -1)
      if np.shape(a)[1] == LD:
          A = np.vstack((A,a))
          name.append(nazwa)

  dataset = sklearn.datasets.base.Bunch(data = A, target=np.array(name))
  return dataset

clf = svm.SVC(gamma=0.001, C=100)
X, y = dataset.data[:-10], dataset.target[:-10]
clf.fit(X, y)
# print(clf.predict(dataset.data[-2]))

dd = czytaj_plik('/home/khrees/testk2.wav')
for i in range(0, len(dd.data)):
  print(clf.predict(dd.data[i]))
