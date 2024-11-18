import numpy as np

class Signal:

    def __init__(self, values, start_index=0):
        # Преобразуем значения в формат с плавающей точкой для точных вычислений
        self.values = [float(v) for v in values]
        self.start_index = start_index
        self.end_index = start_index + len(values) - 1
        self.length = len(values)

    def __repr__(self):
        return f"Signal(values={self.values}, start_index={self.start_index}, end_index={self.end_index}, length={self.length})"

    # Сложение двух сигналов
    def __add__(self, other):
        if self.length != other.length:
            raise ValueError("Сигналы для сложения должны иметь одинаковую длину.")
        result = [x + y for x, y in zip(self.values, other.values)]
        return Signal(result, min(self.start_index, other.start_index))

    # Умножение сигнала на число
    def __mul__(self, scalar):
        result = [x * scalar for x in self.values]
        return Signal(result, self.start_index)

    # Upsampling
    def upsample(self, factor):
        if factor <= 1:
            raise ValueError("Коэффициент увеличения должен быть больше 0.")

        result = []
        for value in self.values:
            result.append(value)
            result.extend([0] * (factor - 1))

        new_length = len(result)
        new_end_index = self.start_index + new_length - 1
        return Signal(result, self.start_index)

    # Downsampling
    def downsample(self, factor):
        if factor <= 1:
            raise ValueError("Коэффициент уменьшения должен быть больше 0.")

        result = self.values[::factor]
        new_length = len(result)
        new_end_index = self.start_index + new_length - 1
        return Signal(result, self.start_index)

    # Прямое преобразование Хаара
    def haar_transform(self):
        n = len(self.values)
        if n & (n - 1) != 0:
            raise ValueError("Длина сигнала должна быть степенью двойки для применения Хаара.")

        output = np.copy(self.values).astype(float)
        h = 1

        # Прямое преобразование Хаара
        while h < len(output):
            for i in range(0, len(output), h * 2):
                avg = (output[i] + output[i + h]) / 2.0
                diff = (output[i] - output[i + h]) / 2.0
                output[i] = avg
                output[i + h] = diff
            h *= 2

        return Signal(list(output), self.start_index)

    # Обратное преобразование Хаара
    def inverse_haar_transform(self):
        n = len(self.values)
        output = np.copy(self.values).astype(float)
        h = n // 2

        # Обратное преобразование Хаара
        while h > 0:
            for i in range(0, len(output), h * 2):
                avg = output[i]
                diff = output[i + h]
                output[i] = avg + diff
                output[i + h] = avg - diff
            h //= 2

        return Signal(list(output), self.start_index)


# Свертка двух сигналов
def convolve(signal1, signal2):
    matrix = []
    result_length = signal1.length + signal2.length - 1
    result_values = [0] * result_length

    for val in signal2.values:
        matrix.append([val * item for item in signal1.values])

    rows, cols = len(matrix), len(matrix[0])

    for sum_indices in range(rows + cols - 1):
        for i in range(max(0, sum_indices - cols + 1), min(sum_indices + 1, rows)):
            j = sum_indices - i
            result_values[sum_indices] += matrix[i][j]

    result_start_index = signal1.start_index + signal2.start_index
    result_end_index = result_start_index + result_length - 1

    return Signal(result_values, result_start_index), result_start_index, result_end_index


signal1 = Signal([1, 2, 3], start_index=5)
signal2 = Signal([0, 1, 0.5], start_index=1)

added_signal = signal1 + signal2
print(f"Сложение сигналов: {added_signal}")

multiplied_signal = signal1 * 2
print(f"Умножение сигнала на 2: {multiplied_signal}")

# Upsampling сигнала
upsampled_signal = signal1.upsample(3)
print(f"Upsampled сигнал: {upsampled_signal}")

# Downsampling сигнала
downsampled_signal = signal1.downsample(2)
print(f"Downsampled сигнал: {downsampled_signal}")

convolved_signal, start_index, end_index = convolve(signal1, signal2)
print(f"Свертка сигналов: {convolved_signal}")
print(f"Стартовый индекс: {start_index}, Конечный индекс: {end_index}")

# Пример использования
signal = Signal([1, 2, 3, 4, 5, 6, 7, 8], start_index=3)
print(f"Исходный сигнал: {signal.values}, {signal.start_index}")

# Прямое преобразование Хаара
haar_transformed_signal = signal.haar_transform()
print(f"Сигнал после преобразования Хаара: {haar_transformed_signal.values}, {signal.start_index}")

# Обратное преобразование Хаара
inverse_haar_signal = haar_transformed_signal.inverse_haar_transform()
print(f"Восстановленный сигнал после обратного преобразования Хаара: {inverse_haar_signal.values}, {signal.start_index}")