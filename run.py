from core import EulerianGrid

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

layer = EulerianGrid(init_data, pressInit=5e6)
layer.run()
print(layer.v_interface[-1])
