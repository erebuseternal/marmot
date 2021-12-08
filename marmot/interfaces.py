import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from IPython.display import clear_output


class ShapeInterface(object):
    def __init__(self, broker, **config):
        self.broker = broker

    def get_target(self, key):
        shape = self.broker.read(key)
        points = np.array(shape.boundary.coords)
        sns.scatterplot(x=points[:,0], y=points[:,1])
        plt.show()
        target = float(input())
        clear_output()
        return target