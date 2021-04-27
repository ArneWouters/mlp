import numpy as np
import matplotlib.pyplot as plt


def draw_graph():
    arr = np.genfromtxt('data/cfr_expl.csv')
    data_points = np.array([i for i in range(arr.size)])
    plt.plot(data_points, arr)
    plt.ylabel('Exploitability')
    plt.xlabel('Iterations')
    plt.legend(['CFR'])
    plt.xlim(left=0)
    # plt.ylim(0, 1)
    # plt.show()
    plt.savefig("images/CFR_expl_graph.png")
    plt.clf()


if __name__ == '__main__':
    draw_graph()
