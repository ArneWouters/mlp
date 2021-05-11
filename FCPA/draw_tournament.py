import numpy as np
import matplotlib.pyplot as plt


def draw_graph():
    model10 = np.array([377250, -1401900, 1914400, 3167850, 470000, 389150, -506450, 2711600, 516300, 1949450])
    model25 = np.array([1832350, -143650, -336250, 1243500, -371350, 1929650, 486300, -1156600, 2830250, 712350])
    model50 = np.array([2610300, 226750, 1959650, -236300, 1615050, 4145500, 2160500, -444500, 646950, 1752250])
    model60 = np.array([-870150, 122200, -198650, -1144700, 1349350, 134750, 1681550, 1109550, 28800, -723700])
    model80 = np.array([1767500, 2899000, 1014250, -1584350, 92600, -1753800, 472950, -1575800, 86050, -339600])
    model100 = np.array([3623800, 1847800, 1571900, 4403400, 2447950, 2577950, 514750, 553500, 3486950, 3455050])
    model120 = np.array([-873600, 1122950, 469250, 1146350, -1621400, 112000, 1962400, 1684850, -716250, 387500])
    model140 = np.array([371100, 2532750, 1237850, 573550, 2802400, 740950, 652750, 903050, 3129300, 1173550])
    randomAgent = np.array([-8838550, -7205900, -7632400, -7569300, -6784600, -8276150, -7424750, -3785650, -10008350, -8366850])

    num_of_tournaments = 10
    x = np.array([i for i in range(1, num_of_tournaments+1)])
    plt.plot(x, model10, label='model10')
    plt.plot(x, model25, label='model25')
    plt.plot(x, model50, label='model50')
    plt.plot(x, model60, label='model60')
    plt.plot(x, model80, label='model80')
    plt.plot(x, model100, label='model100')
    plt.plot(x, model120, label='model120')
    plt.plot(x, model140, label='model140')
    plt.plot(x, randomAgent, label='random')
    plt.legend()
    plt.xticks(np.arange(1, num_of_tournaments + 1, 1.0))
    # plt.xlim(left=0.5)
    plt.xlabel("Tournament")
    plt.ylabel("Winnings")
    plt.show()


if __name__ == '__main__':
    draw_graph()
