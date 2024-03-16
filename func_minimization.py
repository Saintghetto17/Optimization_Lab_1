import math
import typing

from matplotlib.axes import Axes
from prettytable import PrettyTable
from scipy import optimize
from sympy.abc import x, y
import enum
import matplotlib.pyplot as plt
import numpy as np

import warnings

warnings.filterwarnings("ignore")


class FUNCTION(enum.Enum):
    FUNC_1 = x ** 2 + (2 * x - 4 * y) ** 2 + (x - 5) ** 2
    FUNC_2 = x ** 2 + y ** 2 - x * y + 2 * x - 4 * y + 3


class GRADIENT_REGIME(enum.Enum):
    CONSTANT_STEP = 0
    CHANGING_STEP_TERNARY = 1
    CHANGING_STEP_DICHOTOMY = 2


class GRADIENT_NAME(enum.Enum):
    LEARNING_RATE = "LEARNING RATE METHOD"
    CHANGING_RATE_TERNARY = "TERNARY RATE METHOD"
    CHANGING_RATE_DICHOTOMY = "DICHOTOMY RATE METHOD"


function: list[FUNCTION] = [FUNCTION.FUNC_1, FUNCTION.FUNC_2]
GLOBAL_MIN: list[float] = [25 / 2, -1]


def function_value(dot: tuple[float, float], func: FUNCTION) -> float:
    if func == FUNCTION.FUNC_1:
        return dot[0] ** 2 + (2 * dot[0] - 4 * dot[1]) ** 2 + (dot[0] - 5) ** 2
    elif func == FUNCTION.FUNC_2:
        return dot[0] ** 2 + dot[1] ** 2 - dot[0] * dot[1] + 2 * dot[0] - 4 * dot[1] + 3


def gradient(dot: tuple[float, float], func: FUNCTION) -> tuple[float, float]:
    if func == FUNCTION.FUNC_1:
        return 2 * (6 * dot[0] - 8 * dot[1] - 5), -16 * (dot[0] - 2 * dot[1])
    elif func == FUNCTION.FUNC_2:
        return 2 * dot[0] - dot[1] + 2, -dot[0] + 2 * dot[1] - 4


EPS_SEARCH = 0.000001
left_board = -5
right_board = 5


def ternary_search_min(func: typing.Callable[[tuple[float, float]], float],
                       left: float,
                       right: float,
                       grad: tuple[float, float],
                       dot: tuple[float, float]) -> float:
    if right - left < EPS_SEARCH:
        return (left + right) / 2
    a: float = (left * 2 + right) / 3
    b: float = (left + right * 2) / 3
    a_dot: tuple[float, float] = (dot[0] - a * grad[0], dot[1] - a * grad[1])
    b_dot: tuple[float, float] = (dot[0] - b * grad[0], dot[1] - b * grad[1])
    if func(a_dot) < func(b_dot):
        return ternary_search_min(func, left, b, grad, dot)
    else:
        return ternary_search_min(func, a, right, grad, dot)


def dichotomy_search_min(func: typing.Callable[[tuple[float]], float], left: float, right: float,
                         grad: tuple[float, float],
                         dot: tuple[float, float]) -> float:
    if right - left < EPS_SEARCH:
        return (left + right) / 2
    median: float = (left + right) / 2
    a_dot: tuple[float, float] = (dot[0] - (median - EPS_SEARCH) * grad[0], dot[1] - (median - EPS_SEARCH) * grad[1])
    b_dot: tuple[float, float] = (dot[0] - (median + EPS_SEARCH) * grad[0], dot[1] - (median + EPS_SEARCH) * grad[1])
    func_val_1 = func(a_dot)
    func_val_2 = func(b_dot)
    if func_val_1 < func_val_2:
        return dichotomy_search_min(func, left, median, grad, dot)
    else:
        return dichotomy_search_min(func, median, right, grad, dot)


def next_step(prev_dot: tuple[float, float],
              gradient_vector: tuple[float, float],
              regime: GRADIENT_REGIME,
              func: FUNCTION,
              learning_rate: float | None = None) -> tuple[float, float]:
    if regime == GRADIENT_REGIME.CONSTANT_STEP:
        return prev_dot[0] - learning_rate * gradient_vector[0], prev_dot[1] - learning_rate * gradient_vector[1]
    elif regime == GRADIENT_REGIME.CHANGING_STEP_TERNARY:
        step: float = ternary_search_min(lambda xy: function_value(xy, func), left_board, right_board,
                                         gradient_vector, prev_dot)
        return prev_dot[0] - step * gradient_vector[0], prev_dot[1] - step * gradient_vector[1]
    elif regime == GRADIENT_REGIME.CHANGING_STEP_DICHOTOMY:
        step: float = dichotomy_search_min(lambda xy: function_value(xy, func), left_board, right_board,
                                           gradient_vector, prev_dot)
        return prev_dot[0] - step * gradient_vector[0], prev_dot[1] - step * gradient_vector[1]


def normalize(vector_1: tuple[float, float], vector_2: tuple[float, float]) -> float:
    return math.sqrt((vector_1[0] - vector_2[0]) ** 2 + (vector_1[1] - vector_2[1]) ** 2)


# Contract: return value : tuple[0] -> counted value; tuple[1] -> number of iterations
def gradient_descent(initial_dot: tuple[float, float],
                     by_dot_normalize: bool,
                     by_func_normalize: bool,
                     EPS: float,
                     func: FUNCTION,
                     regime: GRADIENT_REGIME,
                     learning_rate: float | None = None) -> tuple[float, int, [list[float], list[float], list[float]]]:
    prev_dot: tuple[float, float] = initial_dot
    prev_func_value: float = 10000000000
    iterations: int = 0
    cord_data = [[], [], []]
    while True:
        iterations += 1
        current_gradient: tuple[float, float] = gradient(prev_dot, func)
        current_dot: tuple[float, float] = next_step(prev_dot, current_gradient, regime, func,
                                                     learning_rate=learning_rate)
        current_func_value: float = function_value(current_dot, func)
        cord_data[0].append(current_dot[0])
        cord_data[1].append(current_dot[1])
        cord_data[2].append(current_func_value)
        if by_dot_normalize:
            if normalize(current_dot, prev_dot) <= EPS:
                return current_func_value, iterations, cord_data
        elif by_func_normalize:
            if abs(current_func_value - prev_func_value) <= EPS:
                return current_func_value, iterations, cord_data
        prev_dot = current_dot
        prev_func_value = current_func_value


######################################################## LEARNING RATE | FAST RATE ##############################################
# Every point will be gone throw every learning rate for analysis
INIT_POINTS: list[tuple[float, float] | tuple[None, None]] = []  # start points
EPS = [0.001, 0.000001, 0.00001]  # EPS at which algorithm will stop
LEARNING_RATES = [0.1, 0.001, 0.001]  # different learning rates
RESULTS: list[list[tuple]] = []  # results tuple(iterations, value)
NUMBER_OF_POINTS = 3

for i in range(NUMBER_OF_POINTS):
    INIT_POINTS.append((i, i))

legend_data = [[], []]


def fill_data(col_names: list[str],
              tables: list[PrettyTable],
              datas: list[list[typing.Any]],
              ax_fig: Axes,
              gradient_name: GRADIENT_NAME, regime: GRADIENT_REGIME,
              numbers_to_display: list[int]) -> None:
    RESULTS = []
    exp_cnt = 0
    for func in range(2):
        RESULTS.append([])
        for i in range(NUMBER_OF_POINTS):
            for j in range(len(EPS)):
                learning_rate = LEARNING_RATES[j]
                try:
                    buffer: tuple[float, int, [list[float], list[float], list[float]]] = ()
                    if regime == GRADIENT_REGIME.CONSTANT_STEP:
                        if j == 2:
                            buffer = (gradient_descent(INIT_POINTS[i], True,
                                                       False, EPS[j], function[func],
                                                       GRADIENT_REGIME.CONSTANT_STEP,
                                                       learning_rate=learning_rate))
                        else:
                            buffer = gradient_descent(INIT_POINTS[i], False,
                                                      True, EPS[j], function[func],
                                                      GRADIENT_REGIME.CONSTANT_STEP,
                                                      learning_rate=learning_rate)
                    elif regime == GRADIENT_REGIME.CHANGING_STEP_TERNARY:
                        buffer = gradient_descent(INIT_POINTS[i], False,
                                                  True, EPS[j], function[func],
                                                  GRADIENT_REGIME.CHANGING_STEP_TERNARY, learning_rate=None)
                    elif regime == GRADIENT_REGIME.CHANGING_STEP_DICHOTOMY:
                        buffer = gradient_descent(INIT_POINTS[i], False, True, EPS[j], function[func],
                                                  GRADIENT_REGIME.CHANGING_STEP_DICHOTOMY, learning_rate=None)
                    RESULTS[func].append(buffer[:2])
                    if exp_cnt in numbers_to_display:
                        l, = ax_fig.plot(buffer[2][0], buffer[2][1], buffer[2][2], '-')
                        ax_fig.scatter(buffer[2][0], buffer[2][1], buffer[2][2])
                        legend_data[0].append(l)
                        legend_data[1].append(gradient_name.value + " " + str(exp_cnt))
                except OverflowError:
                    RESULTS[func].append((None, None))
                exp_cnt += 1
    experiment_number = 0
    for func in range(2):
        tables.append(PrettyTable(col_names))
        datas.append([])
        for i in range(NUMBER_OF_POINTS):
            for j in range(len(EPS)):
                datas[func].append(experiment_number)
                datas[func].append(function[func].value)
                datas[func].append(GLOBAL_MIN[func])
                datas[func].append(INIT_POINTS[i])
                val = RESULTS[func][j + NUMBER_OF_POINTS * i][1]
                datas[func].append(RESULTS[func][j + NUMBER_OF_POINTS * i][1])
                datas[func].append(EPS[j])
                if regime == GRADIENT_REGIME.CONSTANT_STEP:
                    datas[func].append(LEARNING_RATES[j])
                elif gradient_name == GRADIENT_NAME.CHANGING_RATE_TERNARY:
                    datas[func].append(GRADIENT_NAME.CHANGING_RATE_TERNARY.value)
                elif gradient_name == GRADIENT_NAME.CHANGING_RATE_DICHOTOMY:
                    datas[func].append(GRADIENT_NAME.CHANGING_RATE_DICHOTOMY.value)
                datas[func].append(RESULTS[func][j + NUMBER_OF_POINTS * i][0])
                if regime == GRADIENT_REGIME.CONSTANT_STEP:
                    if j == NUMBER_OF_POINTS - 1:
                        datas[func].append(True)
                        datas[func].append(False)
                    else:
                        datas[func].append(False)
                        datas[func].append(True)
                experiment_number += 1


def show_result(cols: list[str], tables: list[PrettyTable], datas: list[list[typing.Any]]):
    number_of_cols: int = len(cols)
    for i in range(len(datas)):
        data: list[typing.Any] = datas[i]
        while data:
            tables[i].add_row(data[:number_of_cols])
            data = data[number_of_cols:]
    for table in tables:
        print(table)
        print()


figure_method_steps = plt.figure()
ax_fms = figure_method_steps.add_subplot(projection='3d')

# VISUALIZATION OF OUR GRAPHICS

x = y = np.arange(-3, 3, 0.05)
X, Y = np.meshgrid(x, y)
Z1 = np.array(function_value((np.ravel(X), np.ravel(Y)), FUNCTION.FUNC_1)).reshape(X.shape)
Z2 = np.array(function_value((np.ravel(X), np.ravel(Y)), FUNCTION.FUNC_2)).reshape(X.shape)

figure = plt.figure(figsize=(14, 10))

ax1 = figure.add_subplot(2, 2, 1, projection='3d')
ax1.plot_surface(X, Y, Z1)
ax2 = figure.add_subplot(2, 2, 2)
cs1 = ax2.contour(X, Y, Z1, level=20)
cs1.clabel()

ax3 = figure.add_subplot(2, 2, 3, projection='3d')
ax3.plot_surface(X, Y, Z2)
ax4 = figure.add_subplot(2, 2, 4)
cs2 = ax4.contour(X, Y, Z2, level=20)
cs2.clabel()

# TABLE FOR LEARNING RATE METHOD
print("################################################# LEARNING RATE ###############################################")
column_names_learning_rate: list[str] = ['№', 'FUNCTION', 'GLOBAL_MIN', 'INIT_POINT', 'ITERATIONS', 'EPS',
                                         'LEARNING_RATE',
                                         'VALUE', 'BY_DOT',
                                         'BY_FUNC']
tables_learning_rate: list[PrettyTable] = []
datas_learning_rate: list[list[typing.Any]] = []
fill_data(column_names_learning_rate, tables_learning_rate, datas_learning_rate,
          ax_fms, GRADIENT_NAME.LEARNING_RATE, GRADIENT_REGIME.CONSTANT_STEP, [7, 9])
show_result(column_names_learning_rate, tables_learning_rate, datas_learning_rate)

# TABLE FOR TERNARY RATE METHOD

print(
    "################################################# CHANGING STEP TERNARY ###############################################")
column_names_ternary_rate: list[str] = ['№', 'FUNCTION', 'GLOBAL_MIN', 'INIT_POINT', 'ITERATIONS', 'EPS',
                                        'CHANGING STEP',
                                        'VALUE']
tables_ternary_rate: list[PrettyTable] = []
datas_ternary_rate: list[list[typing.Any]] = []
fill_data(column_names_ternary_rate, tables_ternary_rate, datas_ternary_rate,
          ax_fms, GRADIENT_NAME.CHANGING_RATE_TERNARY, GRADIENT_REGIME.CHANGING_STEP_TERNARY, [1, 5, 17])
show_result(column_names_ternary_rate, tables_ternary_rate, datas_ternary_rate)

# TABLE FOR NELDER-MEAD:
print(
    "################################################# NELDER-MEAD ##################################################")
column_names_nelder_mead: list[str] = ['FUNCTION', 'GLOBAL_MIN', 'INIT_POINT', 'ITERATIONS',
                                       'VALUE']
tables_nelder_mead: list[PrettyTable] = []
datas_nelder_mead: list[list[typing.Any]] = []

for i in range(2):
    tables_nelder_mead.append(PrettyTable(column_names_nelder_mead))
    datas_nelder_mead.append([])
    for j in range(len(INIT_POINTS)):
        res: optimize.OptimizeResult = optimize.minimize(lambda x: function_value(x, function[i]), INIT_POINTS[j],
                                                         method='nelder-mead')
        datas_nelder_mead[i].append(function[i].value)
        datas_nelder_mead[i].append(GLOBAL_MIN[i])
        datas_nelder_mead[i].append(INIT_POINTS[j])
        datas_nelder_mead[i].append(res.nit)
        datas_nelder_mead[i].append(res.fun)
show_result(column_names_nelder_mead, tables_nelder_mead, datas_nelder_mead)

# TABLE FOR DICHOTOMY : BONUS
print(
    "################################################# BONUS: DICHOTOMY ##############################################")

column_names_dichotomy_rate: list[str] = ['№', 'FUNCTION', 'GLOBAL_MIN', 'INIT_POINT', 'ITERATIONS', 'EPS',
                                          'CHANGING STEP',
                                          'VALUE']
tables_dichotomy_rate: list[PrettyTable] = []
datas_dichotomy_rate: list[list[typing.Any]] = []
fill_data(column_names_dichotomy_rate, tables_dichotomy_rate, datas_dichotomy_rate,
          ax_fms, GRADIENT_NAME.CHANGING_RATE_DICHOTOMY, GRADIENT_REGIME.CHANGING_STEP_DICHOTOMY, [2, 12, 15])
show_result(column_names_dichotomy_rate, tables_dichotomy_rate, datas_dichotomy_rate)

figure_method_steps.legend(legend_data[0], legend_data[1], loc='upper right', shadow=True)
plt.show()
