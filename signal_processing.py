import math
import copy
import sympy

class Signal:
    def __init__(self, values, start_index=0, sig_name=None):
        self.values = values
        self.start_index = start_index
        self.length = len(values)
        self.end_index = self.start_index + self.length - 1
        self.sig_name = sig_name

    def mul(self, scalar):
        scaled_values = [x * scalar for x in self.values]
        return Signal(scaled_values, self.start_index)

    def add(self, other):
        """
        Сложение двух сигналов с учетом их индексов начала и конца.
        :param other: Другой сигнал для сложения.
        :return: Новый сигнал, являющийся суммой двух сигналов.
        """
        new_start = min(self.start_index, other.start_index)
        new_end = max(self.end_index, other.end_index)
        new_length = new_end - new_start + 1
        result_values = [0] * new_length

        # Смещение для первого сигнала
        offset_self = self.start_index - new_start
        for i, val in enumerate(self.values):
            result_values[i + offset_self] += val

        # Смещение для второго сигнала
        offset_other = other.start_index - new_start
        for i, val in enumerate(other.values):
            result_values[i + offset_other] += val

        return Signal(result_values, new_start)

    def convolve(self, filter):
        signal = self.values
        kernel = filter.values

        # Инициализируем пустую матрицу
        matrix = []
        result_length = len(signal) + len(kernel) - 1
        result_values = [0] * result_length

        # Формируем матрицу умножений фильтра и сигнала
        for val in kernel:
            matrix.append([val * item for item in signal])

        # Определяем количество строк и столбцов в матрице
        rows, cols = len(matrix), len(matrix[0])

        # Выполняем свертку путем суммирования по диагоналям
        for sum_indices in range(rows + cols - 1):
            for i in range(max(0, sum_indices - cols + 1), min(sum_indices + 1, rows)):
                j = sum_indices - i
                result_values[sum_indices] += matrix[i][j]

        # Начальный индекс свертки
        start_index = self.start_index + filter.start_index
        return Signal(result_values, start_index)

    def upsample(self, factor):
        """
        Увеличение дискретности сигнала путем добавления нулей между значениями.
        :param factor: Коэффициент увеличения дискретности.
        :return: Upsampled сигнал.
        """
        upsampled_values = []
        for i, value in enumerate(self.values):
            upsampled_values.append(value)
            if i < len(self.values) - 1:
                upsampled_values.extend([0] * (factor - 1))
        return Signal(upsampled_values, self.start_index)

    def downsample(self, factor):
        """
        Уменьшение дискретности сигнала путем выборки каждого factor-го значения.
        :param factor: Коэффициент уменьшения дискретности.
        :return: Downsampled сигнал.
        """
        downsampled_values = []
        for i in range(0, self.length, factor):
            downsampled_values.append(self.values[i])
        return Signal(downsampled_values, self.start_index)

    def get_info(self):
        """
        Вывод информации о сигнале.
        :return: Строка с информацией о сигнале.
        """
        return (f"Signal(values={self.values}, "
                f"start_index={self.start_index}, "
                f"end_index={self.end_index}, "
                f"length={self.length})")

    def __add__(self, other):
        return self.add(other)

    def Analysis(self, h0, h1):
        r0 = self.convolve(h0)
        r1 = self.convolve(h1)
        y0 = r0.downsample(2)
        y1 = r1.downsample(2)
        return y0, y1

    def Synthesis(self, y0,y1,f0,f1):
        t0 = y0.upsample(2)
        t1 = y1.upsample(2)
        v0 = t0.convolve(f0)
        v1 = t1.convolve(f1)
        start_index = min(v0.start_index, v1.start_index)
        v0 = Signal(v0.values, v0.start_index)
        v1 = Signal(v1.values, v1.start_index)
        result_signal = v0 + v1

        # Убираем нулевые значения только если они появились искусственно
        if result_signal.start_index < 0:
            # Проверка для искусственных нулей
            while result_signal.values[0] == 0 and result_signal.start_index < 0:
                result_signal.values.pop(0)
                result_signal.start_index += 1

        original_length = self.length
        if len(result_signal.values) > original_length:
            result_signal.values = result_signal.values[:original_length]
            # Обновляем конечный индекс
        result_signal.end_index = result_signal.start_index + len(result_signal.values) - 1

        return result_signal

    def factorize_polynomial(self, poly=None):
        """
        Factorize the signal values as a polynomial.

        Example:
        ```
        signal.factorize_polynomial()
        ```

        Returns:
            sympy.Poly: The factorized polynomial.
        """
        x = sympy.symbols('x')

        # Если полином не передан, строим его из значений сигнала
        if poly is None:
            polynomial = sum(
                value * x ** (index + self.start_index) for index, value in enumerate(self.values)
            )
        else:
            polynomial = poly

        # Приведение к многочлену SymPy и факторизация
        factorized_polynomial = sympy.Poly(polynomial).factor_list()
        return factorized_polynomial

    def generate_filters(self, HF):
        """
        Generate filters H1, F1 using H0, F0.

        Returns:
            Signal: Filter H0.
            Signal: Filter F0.
            Signal: Filter H1.
            Signal: Filter F1.
        """
        if len(HF[1]) != 2:
            raise ValueError("Need H0 and F0")

        c1 = 1
        c2 = 1

        try:
            c1 = HF[0]
        except IndexError:
            pass

        try:
            c2 = HF[2]
        except IndexError:
            pass

        # Ensure the use of the correct variable
        x = sympy.symbols('x')

        H0 = Signal(
            [c1 * coef for coef in sympy.Poly(HF[1][0][0] ** HF[1][0][1], x).all_coeffs()],
            sig_name="H0"
        )
        F0 = Signal(
            [c2 * coef for coef in sympy.Poly(HF[1][1][0] ** HF[1][1][1], x).all_coeffs()],
            sig_name="F0"
        )
        F1 = Signal(
            [coef * (-1) ** (n + 1) for coef, n in zip(H0.values, range(len(H0.values)))],
            sig_name="F1"
        )
        H1 = Signal(
            [coef * (-1) ** n for coef, n in zip(F0.values, range(len(F0.values)))],
            sig_name="H1"
        )
        return H0, F0, H1, F1

    def filter_bank_6th_degree(self, HF):
        """
        Implement a filter bank for a polynomial of 6th degree.

        Example:
        ```
        signal.filter_bank_6th_degree()
        ```

        Returns:
            list: A list of tuples containing filters (H0, F0, H1, F1).
        """
        rest = copy.deepcopy(HF)  # Сохраняем исходное состояние HF
        result = []

        while HF[1][0][1] > 0:  # Пока степень H0 не равна нулю
            filters = self.generate_filters(HF)
            result.append(filters)

            # Обновляем HF для следующей итерации
            HF[1][0] = (HF[1][0][0], HF[1][0][1] - 1)
            HF[1][1] = (HF[1][1][0] * HF[1][0][0], 1)

        # Добавляем последний набор фильтров
        filters = self.generate_filters(HF)
        result.append(filters)

        HF = rest  # Восстанавливаем HF
        return result

    def direct_transform(source_signal):
        if source_signal.length == 1:
            return source_signal

        ret_val = []
        tmp_arr = []

        # Выполняем преобразование
        for i in range(0, source_signal.length - 1, 2):
            avg = (source_signal.values[i] + source_signal.values[i + 1]) / 2.0
            diff = (source_signal.values[i] - source_signal.values[i + 1]) / 2.0
            ret_val.append(diff)
            tmp_arr.append(avg)

        # Рекурсивно применяем к средней части
        tmp_signal = Signal(tmp_arr, source_signal.start_index // 2)
        transformed_tmp = Signal.direct_transform(tmp_signal)

        # Возвращаем объединённый результат
        ret_val.extend(transformed_tmp.values)
        return Signal(ret_val, source_signal.start_index)

    def inverse_transform(source_signal):
        if source_signal.length == 1:
            return source_signal

        mid = source_signal.length // 2
        ret_val = []

        # Извлекаем временную часть
        tmp_part = Signal(
            source_signal.values[mid:],
            source_signal.start_index + mid
        )

        # Рекурсивно применяем обратное преобразование
        second_part = Signal.inverse_transform(tmp_part)

        for i in range(mid):
            avg = second_part.values[i]
            diff = source_signal.values[i]

            ret_val.append(avg + diff)  # Восстановленный первый элемент
            ret_val.append(avg - diff)  # Восстановленный второй элемент

        return Signal(ret_val, source_signal.start_index)

# Создание сигналов
signal_1 = Signal([1, 2, 3], start_index=0)
signal_2 = Signal([4, 5], start_index=-1)

print("Сигнал 1:", signal_1.get_info())
print("Сигнал 2:", signal_2.get_info())

# Сложение сигналов
sum_signal = signal_1.add(signal_2)
print("Сумма сигналов:", sum_signal.get_info())

# Умножение на число
mul_signal = signal_1.mul(2)
print("Сигнал 1, умноженный на скаляр:", mul_signal.get_info())

# Свертка сигналов
kernel = Signal([0.5, 1], start_index=0)
convolved_signal = signal_1.convolve(kernel)
print("Свертка сигнала 1 и ядра:", convolved_signal.get_info())

#downsampled_signal = signal_1.downsample(2)
#print(downsampled_signal.get_info())

#upsampled_signal = signal_1.upsample(2)
#print(upsampled_signal.get_info())

# Добавляем фильтры для анализа и синтеза
sq = math.sqrt(2)
h0 = Signal([1/sq, 1/sq], start_index=0)  # фильтр для анализа
h1 = Signal([1/sq, -1/sq], start_index=0)  # другой фильтр для анализа
f0 = Signal([1/sq, 1/sq], start_index=-1)  # фильтр для синтеза
f1 = Signal([-1/sq, 1/sq], start_index=-1)  # другой фильтр для синтеза

# Проведение анализа
y0, y1 = signal_1.Analysis(h0, h1)
print("Анализ:")
print("y0:", y0.get_info())
print("y1:", y1.get_info())

# Проведение синтеза
synthesized_signal = signal_1.Synthesis(y0, y1, f0, f1)
print("Синтезированный сигнал:", synthesized_signal.get_info())

signal = Signal([1, 2, 3, 4, 5, 6, 7, 8], start_index=0)

# Пример факторизации
factorized = signal.factorize_polynomial()
print("Факторизация:", factorized)

x = sympy.symbols('x')
poly = x**7 - 3*x**6 + 3*x**5 - x**4

# Факторизация
factorized = sympy.Poly(poly).factor_list()
print("Факторизация2:", factorized)

# Генерация фильтров
H0, F0, H1, F1 = signal.generate_filters((1, [(sympy.symbols('x') + 1, 3), (sympy.symbols('x') - 1, 2)]))
print("H0:", H0.values, "F0:", F0.values, "H1:", H1.values, "F1:", F1.values)

# Фильтрация 6-й степени
filters = signal.filter_bank_6th_degree((1, [(sympy.symbols('x') + 1, 6), (sympy.symbols('x') - 1, 6)]))
# print("Filters:", [[f.values for f in step] for step in filters])
for i, (H0, F0, H1, F1) in enumerate(filters):
        print(f"Шаг {i + 1}:")
        print(f"H0: {H0.values}")
        print(f"F0: {F0.values}")
        print(f"H1: {H1.values}")
        print(f"F1: {F1.values}")

# Прямое преобразование
transformed_signal = Signal.direct_transform(signal)
print("Прямое преобразование:", transformed_signal.get_info())

# Обратное преобразование
restored_signal = Signal.inverse_transform(transformed_signal)
print("Обратное преобразование:", restored_signal.get_info())