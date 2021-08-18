import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import skfuzzy.control as ctrl
from sklearn.metrics import mean_squared_error

def simulate(simulation, xi):
    simulation.input['A'] = xi
    simulation.compute()
    print('z = ',simulation.output['B'])

x = np.arange(-2, 2, 0.01, )
y = x ** 2

a = ctrl.Antecedent(x, "A")
a['a1'] = fuzz.trimf(x, [-2,-2,-1.25])
a['a2'] = fuzz.trimf(x, [-1.75,-1,-1])
a['a3'] = fuzz.trimf(x, [-1,-0.75,-0.5])
a['a4'] = fuzz.trimf(x, [-0.75,-0.5,-0.25])
a['a5'] = fuzz.trimf(x, [-0.5,0,0.5])
a['a6'] = fuzz.trimf(x, [0.25,0.5,0.75])
a['a7'] = fuzz.trimf(x, [0.5,0.75,1])
a['a8'] = fuzz.trimf(x, [1,1,1.75])
a['a9'] = fuzz.trimf(x, [1.25,2,2])

b = ctrl.Consequent(np.arange(y.min(), y.max(), 0.01), "B")
b['b1'] = fuzz.trimf(b.universe, [0,0,0.25])
b['b2'] = fuzz.trimf(b.universe, [0.0625,0.25,0.5625])
b['b3'] = fuzz.trimf(b.universe, [0.25,0.5625,1])
b['b4'] = fuzz.trimf(b.universe, [1,1,3.0625])
b['b5'] = fuzz.trimf(b.universe, [1.5625,4,4])

r1 = ctrl.Rule(antecedent=a['a1'], consequent=b['b5'], label='R1')
r2 = ctrl.Rule(antecedent=a['a2'], consequent=b['b4'], label='R2')
r3 = ctrl.Rule(antecedent=a['a3'], consequent=b['b3'], label='R3')
r4 = ctrl.Rule(antecedent=a['a4'], consequent=b['b2'], label='R4')
r5 = ctrl.Rule(antecedent=a['a5'], consequent=b['b1'], label='R5')
r6 = ctrl.Rule(antecedent=a['a6'], consequent=b['b2'], label='R6')
r7 = ctrl.Rule(antecedent=a['a7'], consequent=b['b3'], label='R7')
r8 = ctrl.Rule(antecedent=a['a8'], consequent=b['b4'], label='R8')
r9 = ctrl.Rule(antecedent=a['a9'], consequent=b['b5'], label='R9')

control_system = ctrl.ControlSystem(rules = [r1, r2, r3, r4, r5, r6, r7, r8, r9])
simulation = ctrl.ControlSystemSimulation(control_system)

simulate(simulation, 0)