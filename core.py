"""
    core.py -- модуль, в котором описан класс 'EulerianGrid' для решения ОЗВБ
        в газодинамической постановке

    Author: Anthony Byuraev 
    Email: toha.byuraev@gmail.com
"""

__author__ = 'Anthony Byuraev'

import numpy as np


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

    def __str__(self):
        return 'Объект класса EulerianGrid'

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
        self.L = init_data.get('L', \
            kwargrs.get('L', self.defaultChar['L']))

        # параметры газа
        self.R = init_data.get('R', self.defaultChar['R'])
        self.k = init_data.get('k', self.defaultChar['k'])

        self.energy_cell = np.full(self.num_coor, self.press / (self.k - 1) / self.ro)
        self.c_cell = np.full(self.num_coor, np.sqrt(self.k * self.press / self.ro))
        self.ro_cell = np.full(self.num_coor, self.ro)
        self.v_cell = np.zeros(self.num_coor)
        self.press_cell = np.zeros(self.num_coor)

        # Для расчета Маха на интерфейсе
        self.mah_cell_m = np.zeros(self.num_coor - 1)
        self.mah_cell_p = np.zeros(self.num_coor - 1)

        # Для расчета потока f (Векторы Ф)
        self.F_param_p = np.array([np.zeros(self.num_coor - 1),
                                   np.zeros(self.num_coor - 1), 
                                   np.zeros(self.num_coor - 1)])
        self.F_param_m = np.array([np.zeros(self.num_coor - 1),
                                   np.zeros(self.num_coor - 1), 
                                   np.zeros(self.num_coor - 1)])
        
        self.c_interface = np.zeros(self.num_coor - 1)
        self.mah_interface = np.zeros(self.num_coor - 1)
        self.press_interface = np.zeros(self.num_coor - 1)
        self.v_interface = np.zeros(self.num_coor - 1)
        self.x_interface = np.zeros(self.num_coor - 1)

        self.f_param = np.array([np.zeros(self.num_coor - 1),
                                 self.press_cell[1:], 
                                 np.zeros(self.num_coor - 1)])
        self.q_param = np.array([self.ro_cell, 
                                 self.ro_cell * self.v_cell,
                                 self.ro_cell * (self.energy_cell + self.v_cell ** 2 / 2)])
        
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

    def run(self):
        """
            Метод - решение. Последовательное интегрирование параметров системы.
        """
        
        # Вычисление координат границ в начальный момент времени
        self._new_x_interf(self.L0)

        # Формирование массивов результатов
        self.coordShell = []        # положение снаряда
        self.velShell = []          # скорость снаряда
        self.pressBottomShell = []  # давление на дно снаряда
        self.pressBottomStem = []   # давление на дно ствола
        self.time = []              # время

        # Последовательное вычисления с шагом по времени
        while True:
            self._get_tau()
            self.x_prev = self.x_interface[1]
            self._new_x_interf(self._end_vel_x()[1])
            self.v_interface[-1] = self._end_vel_x()[0]
            # линейное распределение скорости
            self.v_interface = ( self.v_interface[-1] / self.x_interface[-1] ) * self.x_interface

            # заполнение массивов для графиков
            if len(self.time) == 0:
                self.time.append(self.tau)
            else:
                buf = self.tau + self.time[len(self.time) - 1]
                self.time.append(buf)
            self.coordShell.append(self.x_interface[-1])
            self.velShell.append(self.v_interface[-1])
            self.pressBottomShell.append(self.press_cell[-2])
            self.pressBottomStem.append(self.press_cell[1])

            # последовательные вычисления
            self._get_c_interface()
            self._get_mah_mp()
            self._get_mah_press_interface()
            self._get_F_mines()
            self._get_F_plus()
            self._get_f()
            self._get_q()
            if self.x_interface[-1] >= self.L:
                break

if __name__ == '__main__':
    print('Error: попытка запустить coreP.py')
