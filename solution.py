"""
    solution.py -- модуль, в котором описан класс 'EulerianGrid' для решения ОЗВБ
        в газодинамической постановке

    Author: Anthony Byuraev 
    Email: toha.byuraev@gmail.com
"""

__author__ = 'Anthony Byuraev'

import numpy as np
import matplotlib.pyplot as plt

init_data = {
    # Для тестовой задачи
    'num_coor': 100,
    'press': 5 * (10 ** 6),
    'ro': 141.471,
    'L0': 0.5,
    'd': 0.03,
    'L': 2,
    'mass': 0.1,
    'k': 1.4,
    'Ku': 0.5,
    'R': 287
}

class EulerianGrid(object):
    """
        Класс, содержащий решение основной задачи внутренней баллистики (пневматика) 
            в газодинамической постановке на подвижной сетке по методу Эйлера

        Parameters:
            init_data {dict} -- словарь с начальными данными хар-к АО

        Keyword arguments:
            pressInit {float} -- начальное давление в каморе
            L0 {float} -- начальная длина запоршневого пространства

    """

    # Параметры АО и газа по умолчанию
    defaultChar = {
        'numCoor': 100,
        'pressInit': 5e6,
        'ro': 141.471,
        'L0': 0.5,
        'd': 0.03,
        'L': 2,
        'shellMass': 0.1,
        'k': 1.4,
        'Ku': 0.5,
        'R': 287
    }

    def __init__(self, init_data, **kwargrs):
        self.num_coor = init_data.get('numCoor', self.defaultChar['numCoor'])
        self.press = init_data.get('pressInit', \
            kwargrs.get('pressInit', self.defaultChar['pressInit']))
        self.ro = init_data.get('ro', self.defaultChar['ro'])
        self.kurant = init_data.get('Ku', self.defaultChar['Ku'])
        self.shellMass = init_data.get('shellMass', self.defaultChar['shellMass'])
        self.tau = 0
        # длина ячейки на пред. шаге (коор-та 1 границы)
        #   необходимо для расчета веторов q
        self.x_prev = 0
        self.S = np.pi * init_data['d'] ** 2 / 4
        self.L0 = init_data.get('L0', \
            kwargrs.get('L0', self.defaultChar['L0']))

        # параметры газа
        self.R = init_data.get('R', self.defaultChar['R'])
        self.k = init_data.get('k', self.defaultChar['k'])

        self.ro_cell = np.full(self.num_coor, self.ro)
        self.v_cell = np.full(self.num_coor, 0.0)
        self.energy_cell = np.full(self.num_coor, self.press / (self.k - 1) / self.ro)
        self.press_cell = np.full(self.num_coor, self.press)
        self.c_cell = np.full(self.num_coor, np.sqrt(self.k * self.press / self.ro))

        # Для расчета Маха на интерфейсе
        self.mah_cell_m = np.full(self.num_coor - 1, 0.0)
        self.mah_cell_p = np.full(self.num_coor - 1, 0.0)

        # Для расчета потока f (Векторы Ф )
        self.F_param_p = np.array([np.full(self.num_coor - 1, 0.0), \
            np.full(self.num_coor - 1, 0.0), np.full(self.num_coor - 1, 0.0)])
        self.F_param_m = np.array([np.full(self.num_coor - 1, 0.0), \
            np.full(self.num_coor - 1, 0.0), np.full(self.num_coor - 1, 0.0)])
        
        self.c_interface = np.full(self.num_coor - 1, 0.0)
        self.mah_interface = np.full(self.num_coor - 1, 0.0)
        self.press_interface = np.full(self.num_coor - 1, 0.0)
        self.v_interface = np.full(self.num_coor - 1, 0.0)
        self.x_interface = np.full(self.num_coor - 1, 0.0)

        self.f_param = np.array([np.full(self.num_coor - 1, 0.0), \
            self.press_cell[1:], np.full(self.num_coor - 1, 0.0)])
        self.q_param = np.array([self.ro_cell, self.ro_cell * self.v_cell, self.ro_cell * \
            (self.energy_cell + self.v_cell ** 2 / 2)])
        
    def _get_q(self):
        coef_stretch = self.x_prev / self.x_interface[1]
        self.q_param[0][1:-1] = coef_stretch * (self.q_param[0][1:-1] - self.tau / self.x_prev * \
            (self.f_param[0][1:] - self.f_param[0][:-1]))
        self.q_param[1][1:-1] = coef_stretch * (self.q_param[1][1:-1] - self.tau / self.x_prev * \
            (self.f_param[1][1:] - self.f_param[1][:-1]))
        self.q_param[2][1:-1] = coef_stretch * (self.q_param[2][1:-1] - self.tau / self.x_prev * \
            (self.f_param[2][1:] - self.f_param[2][:-1]))
        self.ro_cell = self.q_param[0]
        self.v_cell = self.q_param[1] / self.q_param[0]
        self.energy_cell = self.q_param[2] / self.q_param[0] - self.v_cell ** 2 / 2
        self.press_cell = self.ro_cell * self.energy_cell * (self.k - 1)
        self.c_cell = np.sqrt(self.k * self.press_cell / self.ro_cell)
        self._border()

    def _get_f(self):
        self.f_param[0] = self.c_interface / 2 * (self.mah_interface * (self.F_param_p[0] + \
            self.F_param_m[0]) - abs(self.mah_interface) * (self.F_param_p[0] - self.F_param_m[0]))
        self.f_param[1] = self.c_interface / 2 * (self.mah_interface * (self.F_param_p[1] + \
            self.F_param_m[1]) - abs(self.mah_interface) * (self.F_param_p[1] - self.F_param_m[1])) + \
                self.press_interface
        self.f_param[2] = self.c_interface / 2 * (self.mah_interface * (self.F_param_p[2] + \
            self.F_param_m[2]) - abs(self.mah_interface) * (self.F_param_p[2] - self.F_param_m[2])) + \
                self.press_interface * self.v_interface

    def _get_F_mines(self):
        self.F_param_m[0] = self.ro_cell[:-1]
        self.F_param_m[1] = self.ro_cell[:-1] * self.v_cell[:-1]
        self.F_param_m[2] = self.ro_cell[:-1] * (self.energy_cell[:-1] + \
            self.v_cell[:-1] ** 2 / 2 + self.press_cell[:-1] / self.ro_cell[:-1])

    def _get_F_plus(self):
        self.F_param_p[0] = self.ro_cell[1:]
        self.F_param_p[1] = self.ro_cell[1:] * self.v_cell[1:]
        self.F_param_p[2] = self.ro_cell[1:] * (self.energy_cell[1:] + \
            self.v_cell[1:] ** 2 / 2 + self.press_cell[1:] / self.ro_cell[1:])

    def _get_c_interface(self):
        self.c_interface = (self.c_cell[1:] + self.c_cell[:-1]) / 2

    def _get_mah_mp(self):
        self.mah_cell_m = (self.v_cell[:-1] - self.v_interface) / self.c_interface
        self.mah_cell_p = (self.v_cell[1:] - self.v_interface) / self.c_interface
    
    def _get_mah_press_interface(self):
        self.mah_interface = self.fetta_plus() + self.fetta_mines()
        self.press_interface = self.getta_plus() * self.press_cell[:-1] + \
            self.getta_mines() * self.press_cell[1:]

    def _get_tau(self):
        buf = (self.x_interface[1:] - self.x_interface[:-1]) / \
            (abs(self.v_cell[1:-1]) + self.c_cell[1:-1])
        self.tau = self.kurant * min(buf)

    def _border(self):
        self.q_param[0][0] = self.q_param[0][1]
        self.q_param[0][self.num_coor - 1] = self.q_param[0][self.num_coor - 2]

        self.v_cell[0] = -self.v_cell[1]
        self.q_param[1][0] = self.ro_cell[0] * self.v_cell[0]
        self.q_param[1][self.num_coor - 1] = self.q_param[0][self.num_coor - 1] * \
            (2 * self.v_interface[self.num_coor - 2] - self.v_cell[self.num_coor - 2])

        self.q_param[2][0] = self.q_param[2][1]
        self.q_param[2][self.num_coor - 1] = self.q_param[2][self.num_coor - 2]

    def _new_x_interf(self, bottomX):
        """
        Метод определяет новые координаты границ
        Parameters:
            bottomX {float} -- координата последней границы
        """
        self.x_interface = np.linspace(0, bottomX, self.num_coor - 1)

    def _end_vel_x(self):
        """
        Функция возвращает список:
            [0] скорость последней границы; [1] коор-та последней границы
        """
        acce = self.press_cell[-2] * self.S / self.shellMass
        speed = self.v_interface[-1] + acce * self.tau
        x = self.x_interface[-1] + self.v_interface[-1] * self.tau + acce * self.tau ** 2 / 2
        return [speed, x]

    def fetta_plus(self):
        """
        Параметр в пересчете давления на границе для индекса +
        """
        # need to fix with vector calculations
        buf = []
        for mah in self.mah_cell_m:
            if abs(mah) >= 1: buf.append(0.5 * (mah + abs(mah)))
            else: buf.append(0.25 * (mah + 1) ** 2 * (1 + 4 * 1 / 8 * (mah - 1) ** 2))
        return np.asarray(buf)

    def fetta_mines(self):
        """
        Параметр в пересчете давления на границе для индекса -
        """
        # need to fix with vector calculations
        buf = []
        for mah in self.mah_cell_p:
            if abs(mah) >= 1: buf.append(0.5 * (mah - abs(mah)))
            else: buf.append(- 0.25 * (mah - 1) ** 2 * (1 + 4 * 1 / 8 * (mah + 1) ** 2))
        return np.asarray(buf)

    def getta_plus(self):
        """
        Параметр в пересчете давления на границе для индекса +
        """
        buf = []
        for mah in self.mah_cell_m:
            if abs(mah) >= 1: buf.append((mah + abs(mah)) / 2 / mah)
            else: buf.append((mah + 1) ** 2 * ((2 - mah) / 4 + 3 / 16 * mah * (mah - 1) ** 2))
        return np.asarray(buf)

    def getta_mines(self):
        """
        Параметр в пересчете давления на границе для индекса -
        """
        buf = []
        for mah in self.mah_cell_p:
            if abs(mah) >= 1: buf.append((mah - abs(mah)) / 2 / mah)
            else: buf.append((mah - 1) ** 2 * ((2 + mah) / 4 - 3 / 16 * mah * (mah + 1) ** 2))
        return np.asarray(buf)


a_time = []
a_speed = []
press_bottom_sn = []
press_bottom_tr = []
a_coord = []

layer = EulerianGrid(init_data, pressInit=5e6)
layer._new_x_interf(init_data['L0'])
while True:
    layer._get_tau()
    layer.x_prev = layer.x_interface[1]
    endSpeed_endX = layer._end_vel_x()
    layer._new_x_interf(endSpeed_endX[1])
    layer.v_interface[-1] = endSpeed_endX[0]
    k_line = layer.v_interface[-1] / endSpeed_endX[1]   #
    layer.v_interface = k_line * layer.x_interface      # Линейное распределение скорости границ

    # заполнение массивов для графиков
    if len(a_time) == 0:
        a_time.append(layer.tau)
    else:
        buf = layer.tau + a_time[len(a_time) - 1]
        a_time.append(buf)
    a_coord.append(layer.x_interface[-1])
    a_speed.append(layer.v_interface[-1])
    press_bottom_sn.append(layer.press_cell[-2])
    press_bottom_tr.append(layer.press_cell[1])

    layer._get_c_interface()
    layer._get_mah_mp()
    layer._get_mah_press_interface()
    layer._get_F_mines()
    layer._get_F_plus()
    layer._get_f()
    layer._get_q()
    if layer.x_interface[-1] >= init_data['L']:
        break

plt.plot(a_time, a_speed)
plt.grid(True)
plt.ylabel('Vp, м/с')
plt.xlabel('t, c')
plt.show()

plt.plot(a_coord, a_speed)
plt.grid(True)
plt.ylabel('Vp, м/с')
plt.xlabel('L, м')
plt.show()

plt.plot(a_time, press_bottom_sn, '--')
plt.plot(a_time, press_bottom_tr)
plt.grid(True)
plt.ylabel('P, Па')
plt.xlabel('t, c')
plt.show()

plt.plot(a_coord, press_bottom_sn, '--')
plt.plot(a_coord, press_bottom_tr)
plt.grid(True)
plt.ylabel('P, Па')
plt.xlabel('L, м')
plt.show()
# print(f'скорость {EulerianGrid.calculate()}')