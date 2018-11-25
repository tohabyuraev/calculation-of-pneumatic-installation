import math as mt
import matplotlib.pyplot as plt

init_data = {
    'num_coor': 100,
    'press': 0.5 * (10 ** 6),
    'ro': 141.741,
    'temp': 293,
    'e': 250,
    'Lo': 0.5,
    'd': 0.03,
    'L': 4,
    'mass': 0.1,
    'k': 1.4,
    'Ku': 0.5,
    'R': 287,
    'atmo': 10 ** 5
}


def get_plot(cordx, cordy, strx, stry):
    """Строит график
    Принимает cordx <float> и тд"""
    plt.plot(cordx, cordy), plt.grid(True), plt.ylabel(stry), plt.xlabel(strx), plt.show()


def fetta(mah, number, string):
    """"Параметр в пересчете чисел Иаха
    Принимает на вход число Маха mah <float>, количество координат number <float> и + или - string <str>"""
    for i in range(number):
        if abs(mah[i]) >= 1 and string == 'plus':
            return 0.5 * (mah[i] + abs(mah[i]))
        if abs(mah[i]) >= 1 and string == 'mines':
            return 0.5 * (mah[i] - abs(mah[i]))
        if abs(mah[i]) < 1 and string == 'plus':
            return 0.25 * ((mah[i] + 1) ** 2) * (1 + 4 * 1 / 8 * ((mah[i] - 1) ** 2))
        if abs(mah[i]) < 1 and string == 'mines':
            return - 0.25 * ((mah[i] - 1) ** 2) * (1 + 4 * 1 / 8 * ((mah[i] + 1) ** 2))


def getta(mah, number, string):
    """Параметр в пересчете давления на границе
    Принимает на вход число Маха mah <float> и + или - string <str>"""
    for i in range(number):
        if abs(mah[i]) >= 1 and string == 'plus':
            return (mah[i] + abs(mah[i])) / 2 / mah[i]
        if abs(mah[i]) >= 1 and string == 'mines':
            return (mah[i] - abs(mah[i])) / 2 / mah[i]
        if abs(mah[i]) < 1 and string == 'plus':
            return ((mah[i] + 1) ** 2) * ((2 - mah[i]) / 4 + 3 / 16 * mah[i] * ((mah[i] - 1) ** 2))
        if abs(mah[i]) < 1 and string == 'mines':
            return ((mah[i] - 1) ** 2) * ((2 + mah[i]) / 4 + 3 / 16 * mah[i] * ((mah[i] + 1) ** 2))


def sp_cr(press_0, press_1, mass, v0, tau):
    """Выводит скорость последней границы и кприращение координаты последней границы
        Входные данные: press-1, press, mass (масса снаряда), vo (скорость снаряда на пред шаге), tau """
    sectional = mt.pi * (init_data['d'] ** 2) / 4
    acce = (press_0 - press_1) * sectional / mass
    speed = v0 + acce * tau
    dx = v0 * tau + acce * (tau ** 2) / 2
    answer_sp_cr = [speed, dx]
    return answer_sp_cr


def coef_stretch(x_now, x_now_next, x_prev, x_prev_next):
    """Функция рассчитывает коэффициент растяжения сетки
        x_now_next это координаты на след шаге по времени [i+1]
        x_prev_next это координаты на наст шаге по времени [i+1]"""
    delta_x_now = 0.5 * (x_now - x_now_next + x_prev - x_prev_next)
    delta_x_prev = x_prev - x_prev_next
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
