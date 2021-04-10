import numpy as np
import matplotlib.pyplot as plt


def draw_graph():
    arr = np.genfromtxt('data/nfsp_expl.csv')
    arr2 = np.genfromtxt('data/pg_a2c_expl.csv')
    arr3 = np.genfromtxt('data/pg_rpg_expl.csv')
    arr4 = np.genfromtxt('data/pg_qpg_expl.csv')
    data_points = np.array([i*10000 for i in range(arr.size)])
    plt.plot(data_points, arr)
    plt.plot(data_points, arr2)
    plt.plot(data_points, arr3)
    plt.plot(data_points, arr4)
    plt.ylabel('Exploitability')
    plt.xlabel('Episodes')
    plt.legend(['NFSP', 'A2C', 'RPG', 'QPG'])
    plt.xlim(left=0)
    # plt.ylim(0, 1)
    # plt.show()
    plt.savefig("images/expl_graph.png")
    plt.clf()


if __name__ == '__main__':
    draw_graph()
