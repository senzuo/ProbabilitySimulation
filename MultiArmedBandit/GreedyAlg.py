import numpy as np
import random
"""
    模拟多臂老虎机
    使用了e贪心算法(greedy函数)和普通函数做对比
    
    但是这个和强化学习有什么关系呢？
    RL中
        Agent即为选择器
        环境为老虎机背后的概率和该选择器感知的概率
        状态为当前的奖励值，只有奖励值的高低
    此例为单步强化学习，比较简单（棋类等为n多步强化学习）
    通过不断的学习，使得长期累积奖赏最大化
"""


def greedy(K=5, T=2000, e=0.1):
    """
    e贪心算法 e的概率探索Exploration 1-e的概率利用Exploitation
    :param K: 摇臂数
    :param T: 尝试次数
    :param e: 探索概率
    :return:r 奖赏
    """
    r = 0
    Q = np.zeros(K)
    count = np.zeros(K)
    for i in range(T):
        if random.random() < e:
            k = random.randint(0, K-1)
        else:
            k = Q.argmax()
        v = reward(k)
        r += v
        Q[k] = (Q[k] * count[k] + v) / (count[k] + 1)
        count[k] += 1
    return r


def normal(K=5, T=2000):
    r = 0
    for i in range(T):
        k = random.randint(0, K-1)
        v = reward(k)
        r += v
    return r


def reward(k):
    """
    老虎机概率
    :param k:编号
    :return:
    """
    # num = [0.3, 0.5, 0.7, 0.1, 0.4]
    num = [0.1, 0.9, 0.4, 0.6, 0.5]
    if random.random() > num[k]:
        return 1
    return 0


if __name__ == '__main__':
    gre = 0
    nor = 0
    for i in range(10):
        g = greedy()
        gre += g
        print(g, end='\t')
        n = normal()
        nor += n
        print(n)
    print('value of greedy is % i', gre)
    print('value of normal is % i', nor)
