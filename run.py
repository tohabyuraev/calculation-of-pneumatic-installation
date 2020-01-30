from core import EulerianGrid
import matplotlib.pyplot as plt

init_data = {
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
