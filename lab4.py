from math import sqrt, fabs as fab
from numpy.linalg import det
from random import randrange
from _pydecimal import Decimal
from scipy.stats import f, t

# Значення за варіантом
m = 0
N = 2 ** 3

min_x1 = -10
max_x1 = 50
min_x2 = 20
max_x2 = 60
min_x3 = -10
max_x3 = 10

min_y = 200 + int((min_x1 + min_x2 + min_x3) / 3)
max_y = 200 + int((max_x1 + max_x2 + max_x3) / 3)

matrixExp = [
    [-1, -1, -1, +1, +1, +1, -1],
    [-1, -1, +1, +1, -1, -1, +1],
    [-1, +1, -1, -1, +1, -1, +1],
    [-1, +1, +1, -1, -1, +1, -1],
    [+1, -1, -1, -1, -1, +1, +1],
    [+1, -1, +1, -1, +1, -1, -1],
    [+1, +1, -1, +1, -1, -1, -1],
    [+1, +1, +1, +1, +1, +1, +1],
]

a1, a2, a3, a11, a22, a33, a12, a13, a23 = 0, 0, 0, 0, 0, 0, 0, 0, 0

def matrixGenerator():
    # Генеруємо матрицю
    matrix_y = [[randrange(min_y, max_y)
                 for y in range(m)] for x in range(N)]
    return matrix_y


def middleValue(arr, orientation):
    # Функція пошуку середнього значення по колонках або по рядках
    middle = []
    if orientation == 1:
        # Середнє значення по рядку
        for rows in range(len(arr)):
            middle.append(sum(arr[rows]) / len(arr[rows]))
    else:
        # Середнє значення по колонкі
        for column in range(len(arr[0])):
            numberArr = []
            for rows in range(len(arr)):
                numberArr.append(arr[rows][column])
            middle.append(sum(numberArr) / len(numberArr))
    return middle


def studentTest(b_lst, number_x=4):
    # Критерій Стьюдента
    dispersion_b = sqrt(sum(dispersion_y) / (N * N * m))
    t_lst = [0.0 for x in range(N)]
    for k in range(number_x):
        for x in range(N):
            if k == 0:
                t_lst[x] += middle_y[x] / N
            else:
                t_lst[x] += middle_y[x] * matrixExp[x][k - 1] / N
    for i in range(len(t_lst)):
        t_lst[i] = fab(t_lst[i]) / dispersion_b
    tt = CritValues.studentValue(f3, q)
    for i in range(number_x):
        if t_lst[i] > tt:
            continue
        else:
            t_lst[i] = 0
    for j in range(number_x):
        b_lst[j] = 0 if t_lst[j] == 0 else b_lst[j]
    return b_lst


def fisherTest(b_lst, number=3):
    # Критерій Фішера
    dispersion_ad = 0
    for i in range(N):
        yj = b_lst[0]
        for j in range(number):
            yj += matrix[i][j] * b_lst[j + 1]
        dispersion_ad += (middle_y[i] - yj) ** 2
    dispersion_ad /= m / (N - d)
    Fp = dispersion_ad / (sqrt(sum(dispersion_y) / (N * N * m)))
    Ft = CritValues.fisherValue(f3, f4, q)
    return True if Fp < Ft else False

class CritValues:
    # Критичні значення
    @staticmethod
    def cohrenValue(selectionSize, selectionQty, significance):
        selectionSize += 1
        partResult1 = significance / (selectionSize - 1)
        params = [partResult1, selectionQty, (selectionSize - 1 - 1) * selectionQty]
        fisher = f.isf(*params)
        result = fisher / (fisher + (selectionSize - 1 - 1))
        return Decimal(result).quantize(Decimal('.0001')).__float__()

    @staticmethod
    def studentValue(f3, significance):
        # Значення критерію Стьюдента
        return Decimal(abs(t.ppf(significance / 2, f3))).quantize(Decimal('.0001')).__float__()

    @staticmethod
    def fisherValue(f3, f4, significance):
        # Значення критерію Фішера
        return Decimal(abs(f.isf(significance, f4, f3))).quantize(Decimal('.0001')).__float__()


correctInp = False
while not correctInp:
    try:
        m = int(input("Кількість повторень: "))
        p = float(input("Довірча ймовірність: "))
        correctInp = True
    except ValueError:
        pass

matrix_x, matrix_3x = [[] for x in range(N)], [[] for x in range(N)]

for i in range(len(matrix_x)):
    x1 = min_x1 if matrixExp[i][0] == -1 else max_x1
    x2 = min_x2 if matrixExp[i][1] == -1 else max_x2
    x3 = min_x3 if matrixExp[i][2] == -1 else max_x3
    matrix_x[i] = [x1, x2, x3, x1 * x2, x1 * x3, x2 * x3, x1 * x2 * x3]
    matrix_3x[i] = [x1, x2, x3]

adequacy, odinority = False, False
# Адекватність і однорідність по замовчуванні False
while not adequacy:
    while not odinority:
        matrix_y = matrixGenerator()
        middle_x = middleValue(arr=matrix_3x, orientation=0)
        middle_y = middleValue(arr=matrix_y, orientation=1)

        for i in range(N):
            a1 += matrix_x[i][0] * middle_y[i] / N
            a2 += matrix_x[i][1] * middle_y[i] / N
            a3 += matrix_x[i][2] * middle_y[i] / N
            a11 += matrix_x[i][0] ** 2 / N
            a22 += matrix_x[i][1] ** 2 / N
            a33 += matrix_x[i][2] ** 2 / N
            a12 += matrix_x[i][0] * matrix_x[i][1] / N
            a13 += matrix_x[i][0] * matrix_x[i][2] / N
            a23 += matrix_x[i][1] * matrix_x[i][2] / N
        a21 = a12
        a31 = a13
        a32 = a23
        my = sum(middle_y) / len(middle_y)
        numb0 = [[my, middle_x[0], middle_x[1], middle_x[2]], [a1, a11, a12, a13], [a2, a21, a22, a23],
                 [a3, a31, a32, a33]]
        numb1 = [[1, my, middle_x[1], middle_x[2]], [middle_x[0], a1, a12, a13], [middle_x[1], a2, a22, a23],
                 [middle_x[2], a3, a32, a33]]
        numb2 = [[1, middle_x[0], my, middle_x[2]], [middle_x[0], a11, a1, a13], [middle_x[1], a21, a2, a23],
                 [middle_x[2], a31, a3, a33]]
        numb3 = [[1, middle_x[0], middle_x[1], my], [middle_x[0], a11, a12, a1], [middle_x[1], a21, a22, a2],
                 [middle_x[2], a31, a32, a3]]
        dividerB = [[1, middle_x[0], middle_x[1], middle_x[2]], [middle_x[0], a11, a12, a13],
                    [middle_x[1], a21, a22, a23], [middle_x[2], a31, a32, a33]]

        b0 = det(numb0) / det(dividerB)
        b1 = det(numb1) / det(dividerB)
        b2 = det(numb2) / det(dividerB)
        b3 = det(numb3) / det(dividerB)

        matrix = []
        for i in range(N):
            matrix.append(matrix_3x[i] + matrix_y[i])

        print("\nМатриця з натуральних значень факторів")
        print("  X1 X2 X3 Y1  Y2  Y3  ")
        for i in range(len(matrix)):
            print("", end=" ")
            for j in range(len(matrix[i])):
                print(matrix[i][j], end=" ")
            print("")

        print("\nРівняння регресії")
        print("{:.3f} + {:.3f} * X1 + {:.3f} * X2 + {:.3f} * X3 = ŷ".format(b0, b1, b2, b3))

        dispersion_y = [0.0 for x in range(N)]
        for i in range(N):
            dispersion_i = 0
            for j in range(m):
                dispersion_i += (matrix_y[i][j] - middle_y[i]) ** 2
            dispersion_y.append(dispersion_i / (m - 1))
        f1 = m - 1
        f2 = N
        f3 = f1 * f2
        q = 1 - p
        Gp = max(dispersion_y) / sum(dispersion_y)
        print("\nКритерій Кохрена")
        Gt = CritValues.cohrenValue(f2, f1, q)
        if Gt > Gp or m >= 25:
            print("Дисперсія однорідна при рівні значимості {:.2f}!\nЗбільшувати m не потрібно.".format(q))
            odinority = True
        else:
            print("Дисперсія не однорідна при рівні значимості {:.2f}!".format(q))
            m += 1
        if m == 25:
            exit()

    print("\nКритерій Стьюдента")
    beta_1 = [b0, b1, b2, b3]
    significant_coefficients = studentTest(beta_1)
    print("{:.3f} + {:.3f} * X1 + {:.3f} * X2 + {:.3f} * X3 = ŷ".format(significant_coefficients[0],
                                                                        significant_coefficients[1],
                                                                        significant_coefficients[2],
                                                                        significant_coefficients[3]))

    d = len(significant_coefficients) - significant_coefficients.count(0)
    f4 = N - d
    print("\nКритерій Фішера")
    if not fisherTest(significant_coefficients):
        print("Рівняння регресії неадекватне стосовно оригіналу\nЕфект взаємодії!")
        beta = [0 for i in range(N)]
        for i in range(N):
            if i == 0:
                beta[i] += sum(middle_y) / len(middle_y)
            else:
                for j in range(7):
                    beta[i] += middle_y[i] * matrixExp[i][j] / N
        print("\nРівняння регресії з ефектом взаємодії")
        print("{:.3f} + {:.3f} * X1 + {:.3f} * X2 + {:.3f} * X3 + {:.3f} * Х1X2 + {:.3f} * Х1X3 + {:.3f} * Х2X3"
              "+ {:.3f} * Х1Х2X3= ŷ".format(beta[0], beta[1], beta[2], beta[3], beta[4], beta[5], beta[6], beta[7]))
        print("\nКритерій Кохрена")
        Gt = CritValues.cohrenValue(f2, f1, q)
        if Gt > Gp or m >= 25:
            print("Дисперсія однорідна при рівні значимості {:.2f}!\nЗбільшувати m не потрібно.".format(q))
            odinority = True
        else:
            print("Дисперсія не однорідна при рівні значимості {:.2f}!".format(q))
            m += 1
        if m == 25:
            exit()
        significant_coefficients = studentTest(beta, 8)
        print("\nКритерій Стьюдента")
        print("{:.3f} + {:.3f} * X1 + {:.3f} * X2 + {:.3f} * X3 + {:.3f} * Х1X2 + {:.3f} * Х1X3 + {:.3f} * Х2X3"
              "+ {:.3f} * Х1Х2X3= ŷ".format(significant_coefficients[0], significant_coefficients[1],
                                            significant_coefficients[2],
                                            significant_coefficients[3],
                                            significant_coefficients[4],
                                            significant_coefficients[5],
                                            significant_coefficients[6],
                                            significant_coefficients[7]))

        d = len(significant_coefficients) - significant_coefficients.count(0)
        f4 = N - d
        if studentTest(beta, 7):
            print("Рівняння регресії адекватне стосовно оригіналу")
            adequacy = True
    else:
        print("Рівняння регресії адекватне стосовно оригіналу")
        adequacy = True