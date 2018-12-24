import math as mt
import matplotlib.pyplot as plt

init_data = {
    # Для тестовой задачи
    'num_coor': 100,
    'press': 5 * (10 ** 6),
    'ro': 141.471,
    'Lo': 0.5,
    'd': 0.03,
    'L': 2,
    'mass': 0.1,
    'k': 1.4,
    'Ku': 0.5,
    'R': 287
}

# init_data = {
#     # Для оптимизации
#     'num_coor': 100,
#     'press': 10 * (10 ** 6),
#     'ro': 12,
#     'Lo': 0.075 * 2,
#     'd': 0.075,
#     'L': 0.075 * 40,
#     'mass': 1,
#     'k': 1.4,
#     'Ku': 0.5,
#     'R': 287
# }

# init_data = {
#     # Для прямой задачи
#     'num_coor': 100,
#     'press': 975 * (10 ** 5),
#     'ro': 12,
#     'Lo': 0.075 * 7,
#     'd': 0.075,
#     'L': 0.075 * 40,
#     'mass': 1,
#     'k': 1.4,
#     'Ku': 0.5,
#     'R': 287
# }

data_test = {
    'Lend': 0.075 * 35,
    'Pressend': 110 * (10 ** 6)
}


def get_plot(cordx, cordy, strx, stry, strtitle):
    """Строит график
    Принимает cordx <float> и тд"""
    plt.plot(cordx, cordy)
    plt.grid(True)
    plt.ylabel(stry)
    plt.xlabel(strx)
    plt.title(strtitle)
    plt.show()


def fetta(mah, string):
    # Функция работает правильно
    """"Параметр в пересчете чисел Маха
    Принимает на вход число Маха mah <float> и + или - string <str>"""
    if abs(mah) >= 1 and string == 'plus':
        return 0.5 * (mah + abs(mah))
    if abs(mah) >= 1 and string == 'mines':
        return 0.5 * (mah - abs(mah))
    if abs(mah) < 1 and string == 'plus':
        return 0.25 * ((mah + 1) ** 2) * (1 + 4 * 1 / 8 * ((mah - 1) ** 2))
    if abs(mah) < 1 and string == 'mines':
        return - 0.25 * ((mah - 1) ** 2) * (1 + 4 * 1 / 8 * ((mah + 1) ** 2))


def getta(mah, string):
    # Функция работает правильно
    """Параметр в пересчете давления на границе
    Принимает на вход число Маха mah <float> и + или - string <str>"""
    if abs(mah) >= 1 and string == 'plus':
        return (mah + abs(mah)) / 2 / mah
    if abs(mah) >= 1 and string == 'mines':
        return (mah - abs(mah)) / 2 / mah
    if abs(mah) < 1 and string == 'plus':
        return ((mah + 1) ** 2) * ((2 - mah) / 4 + 3 / 16 * mah * ((mah - 1) ** 2))
    if abs(mah) < 1 and string == 'mines':
        return ((mah - 1) ** 2) * ((2 + mah) / 4 - 3 / 16 * mah * ((mah + 1) ** 2))


def sp_cr(press_0, mass, v0, x0, tau):
    # Функция работает правильно
    """Выводит скорость последней границы и кприращение координаты последней границы
        Входные данные: press-1, press, mass (масса снаряда), vo (скорость снаряда на пред шаге), tau """
    sectional = mt.pi * (init_data['d'] ** 2) / 4
    acce = press_0 * sectional / mass
    speed = v0 + acce * tau
    x = x0 + v0 * tau + acce * (tau ** 2) / 2
    answer_sp_cr = [speed, x]
    return answer_sp_cr


def coef_stretch(x_now, x_now_next, x_prev, x_prev_next):
    # Функция работает правильно?
    """Функция рассчитывает коэффициент растяжения сетки
        x_now_next это координаты на след шаге по времени [i+1]
        x_prev_next это координаты на наст шаге по времени [i]"""
    delta_x_now = x_now_next - x_now
    delta_x_prev = x_prev_next - x_prev
    coef = delta_x_prev / delta_x_now
    answer = [coef, delta_x_prev]
    return answer


def get_all_value(value, all_value, str):
    """Создает массивы по времени (координаты времени)
     и по скорости (значения скорости)
      и по давлению (значения давления) для построения графиков"""
    if len(all_value) > 0 and str == 'time':
        buf = value + all_value[len(all_value) - 1]
        all_value.append(buf)
    if len(all_value) == 0 and str == 'time':
        all_value.append(value)
    if str == 'speed':
        all_value.append(value)
    if str == 'press':
        all_value.append(value)


def get_kpd(k, mass, speed, press, length0):
    return (k - 1) / 2 * mass * (speed ** 2) / press / (mt.pi * (init_data['d'] ** 2) / 4 * length0)