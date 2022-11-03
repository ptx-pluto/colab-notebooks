from sympy import *

t, g, l1, l2, m1, m2, ddq1, ddq2 = symbols('t g l_1 l_2 m_1 m_2 \ddot{q_1} \ddot{q_2}')
q1 = Function('q_1')(t)
q2 = Function('q_2')(t)
dq1 = Function('\dot{q_1}')(t)
dq2 = Function('\dot{q_2}')(t)

x1 = l1 * cos(q1)
y1 = -l1 * sin(q1)
x2 = x1 + l2 * cos(q1 + q2)
y2 = y1 - l2 * sin(q1 + q2)

# substitution rules
rule = [(q1.diff(t), dq1), (q2.diff(t), dq2), (dq1.diff(t), ddq1), (dq2.diff(t), ddq2)]

dx1 = x1.diff(t).subs(rule)
dy1 = y1.diff(t).subs(rule)
dx2 = x2.diff(t).subs(rule)
dy2 = y2.diff(t).subs(rule)

T1 = S.Half * m1 * (dx1 ** 2 + dy1 ** 2)
T2 = S.Half * m2 * (dx2 ** 2 + dy2 ** 2)

V1 = m1 * g * y1
V2 = m2 * g * y2

T = T1 + T2
V = V1 + V2

L = T - V

eq1 = L.diff(dq1).diff(t) - L.diff(q1)
eq1 = eq1.subs(rule)
eq1 = simplify(eq1)

eq2 = L.diff(dq2).diff(t) - L.diff(q2)
eq2 = eq2.subs(rule)
eq2 = simplify(eq2)

ans = solve((Eq(eq1, 0), Eq(eq2, 0)), (ddq1, ddq2))
sol_ddq1 = simplify(ans[ddq1])
sol_ddq2 = simplify(ans[ddq2])

y = MatrixSymbol('y', 4, 1)
theta = MatrixSymbol('theta', 5, 1)

# substitution rules
rules_eom = [
    (m1, theta[0, 0]),
    (m2, theta[1, 0]),
    (l1, theta[2, 0]),
    (l2, theta[3, 0]),
    (g, theta[4, 0]),
    (q1, y[0, 0]),
    (q2, y[1, 0]),
    (dq1, y[2, 0]),
    (dq2, y[3, 0])
]

# right hand side
eom_rhs = ImmutableMatrix([
    y[2, 0],
    y[3, 0],
    sol_ddq1.subs(rules_eom),
    sol_ddq2.subs(rules_eom)
])
