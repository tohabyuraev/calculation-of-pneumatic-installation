import numpy as np
from Modul_1 import *


class EulerianGrid(object):
    def __init__(self, init_data):
        self.buf = 0
        self.tau = 0
        self.temp = init_data['temp']
        self.R = init_data['R']
        self.k = init_data['k']
        self.kurant = init_data['Ku']
        self.num_coor = init_data['num_coor']

        self.press_cell = np.full(init_data['num_coor'], init_data['press'])
        self.ro_cell = np.full(init_data['num_coor'], init_data['ro'])
        self.v_cell = np.full(init_data['num_coor'], 0)
        self.energy_cell = np.full(init_data['num_coor'], init_data['press'] / (init_data['k'] - 1) / init_data['ro'])
        self.c_cell = np.full(init_data['num_coor'], np.sqrt(init_data['k'] * init_data['press'] / init_data['ro']))
        # Для расчета Маха на интерфейсе
        self.mah_cell_m = np.full(init_data['num_coor'], 0)
        self.mah_cell_p = np.full(init_data['num_coor'], 0)
        # Для расчета потока f (Векторы Ф )
        self.ff_param_m = np.full(init_data['num_coor'], 0)
        self.ff_param_p = np.full(init_data['num_coor'], 0)

        # Первая ячейка является нулевой и их 100
        # Границы располагаются справа и их 100

        self.mah_interface = np.full(init_data['num_coor'], 0)
        self.c_interface = np.full(init_data['num_coor'], 0)
        self.x_interface = np.linspace(0, init_data['Lo'], init_data['num_coor'])
        self.press_interface = np.full(init_data['num_coor'], 0.0)
        self.v_interface = np.full(init_data['num_coor'], 0)

        self.f_param = np.array([self.ro_cell * self.v_cell, self.press_cell + self.ro_cell * (self.v_cell ** 2),
                                self.v_cell * self.ro_cell * ((self.energy_cell + (self.v_cell ** 2) / 2) +
                                                              self.press_cell / self.ro_cell)])
        self.q_param = np.array([self.ro_cell, self.ro_cell * self.v_cell, self.ro_cell *
                                (self.energy_cell + (self.v_cell ** 2) / 2)])

    def get_q(self, x_interface_prev):
        self.buf = coef_stretch(self.x_interface, np.roll(self.x_interface, -1), x_interface_prev,
                                np.roll(x_interface_prev, 1))
        # В переменную buf записаны [0] коэффициент растяжения и [1] расстояние между границами на пред шаге
        self.q_param[0] = self.buf[0] * (self.q_param[0] - self.tau / self.buf[1] * (np.roll(self.q_param[0], -1) -
                                                                                     self.q_param[0]))
        self.q_param[1] = self.buf[0] * (self.q_param[1] - self.tau / self.buf[1] * (np.roll(self.q_param[1], -1) -
                                                                                     self.q_param[1]))
        self.q_param[2] = self.buf[0] * (self.q_param[2] - self.tau / self.buf[1] * (np.roll(self.q_param[2], -1) -
                                                                                     self.q_param[2]))
        self.ro_cell = self.q_param[0]
        # print(self.ro_cell)
        # Плотность при пересчете получается много отрицательной
        self.v_cell = self.q_param[1] / self.ro_cell
        # self.press_cell = self.ro_cell * self.temp * self.R
        # Значение давления выходит за границы типа int
        # self.c_cell = np.sqrt(self.k * self.press_cell / self.ro_cell)
        # return q_param

    def get_f(self):
        # (self.ff_param_p[0] - self.ff_param_m[0]) разность равняется 0, а это не хорошо
        self.f_param[0] = 0.5 * self.c_interface * (self.mah_interface *
                                                    (self.ff_param_p[0] + self.ff_param_m[0]) -
                                                    abs(self.mah_interface) *
                                                    (self.ff_param_p[0] - self.ff_param_m[0]))
        self.f_param[1] = 0.5 * self.c_interface * (self.mah_interface *
                                                    (self.ff_param_p[1] + self.ff_param_m[1]) -
                                                    abs(self.mah_interface) *
                                                    (self.ff_param_p[1] - self.ff_param_m[1])) + self.press_interface
        self.f_param[2] = 0.5 * self.c_interface * (self.mah_interface *
                                                    (self.ff_param_p[2] + self.ff_param_m[2]) -
                                                    abs(self.mah_interface) *
                                                    (self.ff_param_p[2] - self.ff_param_m
                                                     [2])) + self.press_interface * self.v_interface

    def get_ff(self, str):
        # Функция работает возможно правильно (по формуле сходится)
        if str == 'mines':
            self.ff_param_m = [self.ro_cell, self.ro_cell * self.v_cell,
                               self.ro_cell * ((self.energy_cell +
                                                (self.v_cell ** 2) / 2) + self.press_cell / self.ro_cell)]
        if str == 'plus':
            self.ff_param_p = [np.roll(self.ro_cell, -1), np.roll(self.ro_cell, -1) * np.roll(self.v_cell, -1),
                               np.roll(self.ro_cell, -1) * ((np.roll(self.energy_cell, -1) +
                                                            np.roll(self.v_cell, -1) ** 2) / 2) + np.roll(
                                   self.press_cell, -1) / np.roll(self.ro_cell, -1)]

    def get_c_interface(self):
        # Функция работает правильно
        self.c_interface = (np.roll(self.c_cell, -1) + self.c_cell) / 2

    def get_mah_m(self):
        # Функция работает неправильно
        # Присваиваемое выражение считается правильно, но self.mah_cell_m остается нулевым
        # self.v_cell = (np.roll(self.v_interface, -1) + self.v_interface) / 2
        self.mah_cell_m = (self.v_cell - self.v_interface) / self.c_interface

    def get_mah_p(self):
        # Аналогично минусу
        self.mah_cell_p = (np.roll(self.v_cell, -1) - self.v_interface) / self.c_interface

    def get_mah_interface(self):
        # Аналогично минусу
        for i in range(self.num_coor - 1):
            self.mah_interface[i] = fetta(self.mah_cell_m[i], 'plus') + fetta(self.mah_cell_p[i], 'mines')

    def get_press_interface(self):
        # Аналогично минусу
        for i in range(self.num_coor - 1):
            self.press_interface[i] = getta(self.mah_cell_m[i], 'plus') * self.press_cell[i] + \
                               getta(self.mah_cell_p[i], 'mines') * self.press_cell[i+1]

    def get_tau(self):
        # Функция работает правильно
        for i in range(self.num_coor - 1):
            self.buf = (self.x_interface[i+1] - self.x_interface[i]) / (abs(self.v_interface) + self.c_cell)
        self.buf = min(self.buf)
        self.tau = self.kurant * self.buf

    # def border(self):
    #     self.c_interface[0] = self.c_cell[0] + np.sqrt(self.k * self.press_cell[0] / self.ro_cell[0]) / 2
    #     return None


layer = EulerianGrid(init_data)
all_time_arr = []
all_speed_arr = []
all_press_arr = []

while layer.x_interface[layer.num_coor - 1] <= init_data['L']:
    layer.get_tau()
    prev_x_interface = layer.x_interface  # Для расчета q
    answer = sp_cr(layer.press_cell[layer.num_coor - 1], 0, init_data['mass'],
                   layer.v_interface[layer.num_coor - 1], layer.tau)
    # print(None)

    # answer возвращает скорость правой границы и приращение координаты
    # Пересчет скоростей и перемещений границ работает правильно
    x_end = layer.x_interface[layer.num_coor - 1] + answer[1]
    layer.x_interface = np.linspace(0, x_end, layer.num_coor)
    layer.v_interface[layer.num_coor - 1] = answer[0]
    k_line = layer.v_interface[layer.num_coor - 1] / x_end
    layer.v_interface = k_line * layer.x_interface

    get_all_value(layer.tau, all_time_arr, 'time')
    get_all_value(layer.v_interface[len(layer.v_interface) - 1], all_speed_arr, 'speed')
    get_all_value(layer.press_cell[len(layer.press_interface) - 1], all_press_arr, 'press')

    layer.get_c_interface()
    layer.get_mah_interface()
    layer.get_press_interface()
    layer.get_ff('mines')
    layer.get_ff('plus')
    layer.get_f()
    layer.get_q(prev_x_interface)
    # print()

get_plot(all_time_arr, all_speed_arr, 'Время', 'Скорость')
# print(layer.get_tau())

# layer.get_c_interface()
# print(layer.ff_param_p[1] - layer.ff_param_m[1])
# print(layer.q_param)
# print(mt.sqrt(layer.k * init_data['press'] / init_data['ro']))
