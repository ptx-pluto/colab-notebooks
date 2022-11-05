import numpy as np
from sympy import lambdify
import inspect
from dynamics.plottable_system import PlottableSystem
import dynamics.eom_double_pendulum as eom


class DoublePendulum(PlottableSystem):

    def __init__(self, m1, m2, l1, l2):
        self.l1 = l1
        self.l2 = l2
        self.m1 = m1
        self.m2 = m2
        self.g = 9.8
        self.theta = np.array([[self.m1, self.m2, self.l1, self.l2, self.g]]).T
        self.f = lambdify([eom.theta, eom.y], eom.y_dot)
        self.x1 = lambdify([eom.theta, eom.y], eom.x1.subs(eom.rules))
        self.y1 = lambdify([eom.theta, eom.y], eom.y1.subs(eom.rules))
        self.x2 = lambdify([eom.theta, eom.y], eom.x2.subs(eom.rules))
        self.y2 = lambdify([eom.theta, eom.y], eom.y2.subs(eom.rules))

    def eom(self, t, y):
        return self.f(self.theta, y.reshape(self.ydim, 1)).reshape(self.ydim, )

    @property
    def title(self):
        return 'Double Pendulum'

    @property
    def bounding_box(self):
        padding_ratio = 1.2
        radius = self.l1 + self.l2
        return padding_ratio * radius * np.array([-1, 1, -1, 1])

    @property
    def ydim(self):
        return 4

    def y2pt(self, y):
        y = y.reshape((4, 1))
        p1 = np.array([
            self.x1(self.theta, y),
            self.y1(self.theta, y)
        ])
        p2 = np.array([
            self.x2(self.theta, y),
            self.y2(self.theta, y)
        ])
        return np.array([[0, 0], p1, p2])


assert (inspect.isabstract(DoublePendulum) is False)
