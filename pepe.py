from voicebak import voice as importer_rykow

ryk_justynka = importer_rykow('justynka.wav')
ryk_madzia = importer_rykow('madzia.wav')
ryk_pawel = importer_rykow('pawel.wav')
ryk_krzysiu = importer_rykow('krzysiu.wav')

from sklearn import svm
import numpy as np

j0 = ryk_justynka.mfcc[4]
m0 = ryk_madzia.mfcc[4]
p0 = ryk_pawel.mfcc[4]
k0 = ryk_krzysiu.mfcc[4]

# X = np.array([item for sublist in ryk_justynka.mfcc for item in sublist])
X = np.concatenate((j0, m0, p0, k0), axis=0)
Y = np.append(np.zeros(len(j0)), [np.ones(len(m0)), np.ones(len(p0))*2, np.ones(len(k0))*3])

clf = svm.SVC(gamma=1/20, C=1)
svc = clf.fit(X, Y)
# print(svc)

if __name__ == '__main__':
  print(np.mean(clf.predict(ryk_justynka.mfcc[3])))
  print(np.mean(clf.predict(ryk_madzia.mfcc[3])))
  print(np.mean(clf.predict(ryk_pawel.mfcc[3])))
  print(np.mean(clf.predict(ryk_krzysiu.mfcc[3])))