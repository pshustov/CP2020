import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
import sobol_seq

def getSphereVolumeExact(D, R = 1.0):


    #через явную формулу и гамма-функцию из модуля scipy, как и просили в условии
    V = (np.pi ** (D/2) * R ** D) / gamma(1 + D/2)




    """
    Функция вычисляет значение объема D-мерной сферы радиуса R рекурентным методом

    --------
    Аргументы:
    D - int, количество измерений
    R = 1 - float, радиус сферы
    --------
    Функция возвращает:
    V - float, объем сферы
    """


    return V


def getSphereVolumePseudorandom(N, D, R = 1.0):

    def inside(point, R):
        sum = 0
        for i in range(len(point)):
            sum += point[i] ** 2
        result = sum < R ** 2
        return result

    rndm = np.random.RandomState(12345)


    array_shoots = rndm.uniform(low=-1, high=1, size=(N, D))

    hits = 0
    for i in array_shoots:
        if inside(i, R) == True:
            hits += 1

    V = (hits / N) * ((2*R)**D)  #умножаем отношение попаданий к общему числу выстрелов на объем гиперкуба размерности D,
    # стороны 2R, чтобы влезла сфера

    """
    Функция вычисляет значение объема D-мерной сферы радиуса R

    --------
    Аргументы:
    N - int, количество случайных точек
    D - int, количество измерений
    R = 1 - float, радиус сферы
    --------
    Функция возвращает:
    V - float, объем сферы
    """

    return V







def getSphereVolumeQuasirandom(N, D, R = 1.0):

    def inside(point, R):
        sum = 0
        for i in range(len(point)):
            sum += point[i] ** 2
        result = sum < R ** 2
        return result



    array_shoots = sobol_seq.i4_sobol_generate(D, N)


    hits = 0
    for i in array_shoots:
        if inside(i, R) == True:
            hits += 1

    V = (hits / N) * ((2*R)**D)


    """
    Функция вычисляет значение объема D-мерной сферы радиуса R

    --------
    Аргументы:
    N - int, количество случайных точек
    D - int, количество измерений
    R = 1 - float, радиус сферы
    --------
    Функция возвращает:
    V - float, объем сферы
    """

    return V




def main_1(N):

    V_exact_array = []
    eps_pseudo = []

    dimensions = [i for i in range(1, 11)] #построим график для первых 15 измерений (этого будет достаточно чтобы сделать
    # выводы о том какая сетка случайных чисел лучше

    for D in range(1, 11):
        V_exact = getSphereVolumeExact(D, R=1.0)
        V_random = getSphereVolumePseudorandom(N, D, R=1.0)
        V_exact_array.append(V_exact)
        eps_pseudo.append((V_exact - V_random) / V_exact)  # будем считать ошибку по данной формуле, да и график впоследствии
        # не будем рисовать в логарифмическом масштабе, так как и без него прекрасно видно, какой из методов
        # выигрывает, а картинка так получается красивее

    plt.figure()
    plt.title('Exact formula сalculation of volume of the hypersphere')
    plt.grid()
    plt.ylabel('Volume')
    plt.xlabel('number_of_dimensions')
    plt.plot(dimensions, V_exact_array, label='formula calculated volume of hypersphere')
    plt.legend()
    plt.show()

    eps_quasi = []
    for D in range(1, 11):
        V_exact = getSphereVolumeExact(D, R=1.0)
        V_random = getSphereVolumeQuasirandom(N, D, R=1.0)
        eps_quasi.append((V_exact - V_random) / V_exact )  #так же как и в случае с псевдорандомными



    plt.figure()
    plt.title('Relative accuracy - N_dimensions for {} points'.format(N))
    plt.grid()
    plt.ylabel('eps')
    plt.xlabel('number_of_dimensions')
    plt.plot(dimensions, eps_pseudo, label='pseudorandom')
    plt.plot(dimensions, eps_quasi, 'r', label='quasirandom')
    plt.legend()
    plt.show()

    #из графика явно видно, что ошибка на квазислучайных меньше, соответсвтенно их использовать предпочтительнее!

main_1(1000) #накидаем 1000 выстрелов, лектор ставил по дефолту именно так
main_1(100) #также неплохо было бы посмотреть на график для 100 выстрелов, превосходство квазислучайных чисел тут даже виднее,
#график точного объема вылезет второй раз, хотя его мы уже видели при первом вызове функции, прошу за это не карать

def getInitialState(N):

    state = np.random.choice([1, -1], size=(N, N))

    '''
    Функция задает случайное начальное состояние
    ---------
    Аргументы:
    N - int, линейный размер решетки 
    --------
    Функция возвращает:
    state - numpy ndarray of ints, массив состояния системы размера NxN
    '''
    return state


def getDeltaE(i, j, state):
    N = state.shape[0] - 1
    state[i][j] *= -1
    if i != N and j != N:
        sum = state[i-1][j-1] + state[i-1][j] + state[i-1][j+1] + state[i][j-1] + state[i][j+1] + state[i+1][j-1] + state[i+1][j] + state[i+1][j+1]
    else:
        if i == N and j != N:
            sum = state[i - 1][j - 1] + state[i - 1][j] + state[i - 1][j + 1] + state[i][j - 1] + state[i][j + 1] + \
                  state[0][j - 1] + state[0][j] + state[0][j + 1]
        if i != N and j == N:
            sum = state[i - 1][j - 1] + state[i - 1][j] + state[i - 1][0] + state[i][j - 1] + state[i][0] + \
                  state[i + 1][j - 1] + state[i + 1][j] + state[i + 1][0]
        if i == N and j == N:
            sum = state[i - 1][j - 1] + state[i - 1][j] + state[i - 1][0] + state[i][j - 1] + state[i][0] + \
                  state[0][j - 1] + state[0][j] + state[0][0]

    E_post = state[i][j] * sum

    E_prev = (-1) * state[i][j] * sum

    dE = E_post - E_prev
    '''
    Функция расчитывает изменение энергии ячейки (i,j) в случае ее переворота (при этом функция сама не меняет сосотояния state)
    ---------
    Аргументы:
    i - int, адресс ячейки вдоль оси 0
    j - int, адресс ячейки вдоль оси 1
    state - numpy ndarray of ints, массив состояния системы размера NxN
    --------
    Функция возвращает:
    dE - float, изменение энергии
    '''

    return dE


def makeFlip(T, state):
    N = state.shape[0] - 1
    for i in range(N+1):
        for j in range(N+1):
            delta_E = getDeltaE(i, j, state)
            if delta_E < 0:
                state[i][j] *= -1
            if delta_E >= 0:
                coefficient = np.random.choice([1, -1], p=[np.exp(-delta_E/T), 1 - np.exp(-delta_E/T)])
                state[i][j] *= coefficient
    '''
    Функция расчитывает изменение энергии ячейки (i,j) в случае ее переворота (при этом функция сама не меняет сосотояния state)
    ---------
    Аргументы:
    T - float, положительное число, безразмерный коэфициент, характеризующий температуру, равный kT/J
    state - numpy ndarray of ints, массив состояния системы размера NxN
    --------
    Функция возвращает:
    state - numpy ndarray of ints, массив нового состояния системы размера NxN
    '''
    return state


def getEnergy(state):

    N = state.shape[0] - 1
    E = 0

    def Energy(N, i, j, state):
        if i != N and j != N:
            sum = state[i - 1][j - 1] + state[i - 1][j] + state[i - 1][j + 1] + state[i][j - 1] + state[i][j + 1] + \
                  state[i + 1][j - 1] + state[i + 1][j] + state[i + 1][j + 1]
        else:
            if i == N and j != N:
                sum = state[i - 1][j - 1] + state[i - 1][j] + state[i - 1][j + 1] + state[i][j - 1] + state[i][j + 1] + \
                      state[0][j - 1] + state[0][j] + state[0][j + 1]
            if i != N and j == N:
                sum = state[i - 1][j - 1] + state[i - 1][j] + state[i - 1][0] + state[i][j - 1] + state[i][0] + \
                      state[i + 1][j - 1] + state[i + 1][j] + state[i + 1][0]
            if i == N and j == N:
                sum = state[i - 1][j - 1] + state[i - 1][j] + state[i - 1][0] + state[i][j - 1] + state[i][0] + \
                      state[0][j - 1] + state[0][j] + state[0][0]
        E_post = state[i][j] * sum
        return E_post

    for i in range(N + 1):
        for j in range(N + 1):
            E += Energy(N, i, j, state)
    '''
    Функция, рассчитывает значение энергии всей системы
    ---------
    Аргументы:
    state - numpy ndarray of ints, массив состояния системы размера NxN
    --------
    Функция возвращает:
    E - float, значение энергии системы
    '''

    return E

def getMagnetization(state):

    M = np.sum(state)

    '''
    Функция, рассчитывает значение намагниченности всей системы
    ---------
    Аргументы:
    state - numpy ndarray of ints, массив состояния системы размера NxN
    --------
    Функция возвращает:
    M - float, значение намагниченности системы
    '''

    return M
def main_2():
    N       = 10        # размер решетки NxN
    Nt      = 100        # количество точек температуры
    eqSteps = 150        # количество раз выполнения makeFlip для установления равновесия
    steps   = 30         # количество раз выполнения makeFlip для усреднения энергии и намагниченности

    T = np.linspace(0.5, 5, Nt)
    E, M = np.zeros(Nt), np.zeros(Nt)


    for t in range(Nt):
        print("Complete", t / Nt * 100, '%\r', end='')

        Esum = Msum = 0
        state = getInitialState(N)

        for i in range(eqSteps):  # установление статистического равновесия
            makeFlip(T[t], state)

        for i in range(steps):  # суммирование по разным состояниям близким к равновеснсому
            makeFlip(T[t], state)
            Esum += getEnergy(state)
            Msum += getMagnetization(state)

        E[t] = Esum / (steps * N * N)
        M[t] = Msum / (steps * N * N)

    print("Done              \r", end='')


    _, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].scatter(T, E)
    ax[0].set_xlabel("Temperature")
    ax[0].set_ylabel("Energy ")

    ax[1].scatter(T, -abs(M), color='blue')
    ax[1].set_xlabel("Temperature")
    ax[1].set_ylabel("Magnetization ")

    plt.show()

    #Не очень хорошо, но видно явление гистерезиса (или чего-то близкого к нему) Наблюдаемое похоже на поведение ферромагнетиков.
    # На низких температурах высокая намагниченность, потом начинает стремительно падать


main_2()