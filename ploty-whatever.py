from voice import dzwk, dzwp, dzwj, dzwm

import numpy as np
import matplotlib.pyplot as plt

we = dzwk.mfcc[16].T

plt.imshow(we, interpolation='nearest')
plt.gca().invert_yaxis()
plt.xlabel('Próbka')
plt.ylabel('Współczynnik MFCC')
plt.colorbar(orientation='horizontal', fraction=0.046)
plt.show()