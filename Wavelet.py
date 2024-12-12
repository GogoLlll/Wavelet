import copy
import librosa.display
import numpy as np
import math
import sympy
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os

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
        upsampled_values = []
        for i, value in enumerate(self.values):
            upsampled_values.append(value)
            if i < len(self.values) - 1:
                upsampled_values.extend([0] * (factor - 1))
        return Signal(upsampled_values, self.start_index)

    def downsample(self, factor):
        downsampled_values = []
        for i in range(0, self.length, factor):
            downsampled_values.append(self.values[i])
        return Signal(downsampled_values, self.start_index)

    def get_info(self):
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

# Функция для сохранения списка значений в WAV файл
def list_to_wav(file_path, values, sample_rate):
    wavfile.write(file_path, sample_rate, np.array(values, dtype=np.float32))

# Функция для построения спектрограммы
def plot_spectrogram(audio_data, sample_rate, output_path):
    D = librosa.stft(audio_data, n_fft=1024)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    plt.figure(figsize=(12, 8))
    librosa.display.specshow(S_db, x_axis='time', y_axis='log', sr=sample_rate)
    plt.title('Spectrogram')
    plt.colorbar(format="%+2.0f dB")
    plt.savefig(output_path)
    plt.close()

# Функция для работы с полиномами
def create_filter_polynomials(level):
    x = sympy.symbols('x')

    if level == 1:
        H0_poly = 1
        F0_poly = -1 / 16 * (2 - sympy.sqrt(3) - x ** (-1)) * (2 + sympy.sqrt(3) - x ** (-1)) * (1 + x ** (-1))**4
    elif level == 2:
        H0_poly = 1 / 2 * (1 + x(-1))
        F0_poly = -1 / 8 * (2 - sympy.sqrt(3) - x**(-1)) * (2 + sympy.sqrt(3) - x**(-1)) * (1 + x**(-1))**3
    elif level == 3:
        H0_poly = 1 / 2 * (1 + x ** (-1)) * (2 + sympy.sqrt(3) - x ** (-1))
        F0_poly = -1 / 8 * (2 - sympy.sqrt(3) - x ** (-1)) * (1 + x ** (-1)) ** 3
    elif level == 4:
        H0_poly = 1 / 8 * (1 + x ** (-1)) ** 3
        F0_poly = -1 / 2 * (2 + sympy.sqrt(3) - x ** (-1)) * (2 - sympy.sqrt(3) - x ** (-1)) * (1 + x ** (-1))
    elif level == 5:
        H0_poly = (sympy.sqrt(3) - 1) / 4 * sympy.sqrt(2) * (2 + sympy.sqrt(3) - x ** (-1)) * (1 + x ** (-1)) ** 2
        F0_poly = -sympy.sqrt(2) / 4 * (sympy.sqrt(3) - 1) * (2 + sympy.sqrt(3) - x ** (-1)) * (
                    2 - sympy.sqrt(3) - x ** (-1)) * (1 + x ** (-1))
    elif level == 6:
        H0_poly = 1 / 16 * (1 + x ** (-1)) ** 4
        F0_poly = -(2 + sympy.sqrt(3) - x ** (-1)) * (2 - sympy.sqrt(3) - x ** (-1))
    else:
        raise ValueError("Unsupported level")

    # Преобразуем многочлены в список коэффициентов
    H0_coeffs = sympy.Poly(H0_poly, x ** (-1)).all_coeffs()
    F0_coeffs = sympy.Poly(F0_poly, x ** (-1)).all_coeffs()

    # Конвертируем коэффициенты в Signal
    H0 = Signal([float(coef) for coef in H0_coeffs])
    F0 = Signal([float(coef) for coef in F0_coeffs])

    # Генерация дополнительных фильтров H1 и F1
    H1 = Signal([coef * (-1) ** n for n, coef in enumerate(H0.values)])
    F1 = Signal([coef * (-1) ** (n + 1) for n, coef in enumerate(F0.values)])

    return H0, F0, H1, F1


# Основная функция для анализа и синтеза с поддержкой уровня
def transform_audio_with_levels_and_save(audio_path, output_folder, level=3):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Загрузка аудиофайла с помощью librosa
    audio_data, sample_rate = librosa.load(audio_path, sr=None)

    # Создаем экземпляр сигнала
    signal = Signal(list(audio_data), start_index=0)

    # Генерация фильтров на основе выбранного уровня
    H0, F0, H1, F1 = create_filter_polynomials(level)

    # Вывод коэффициентов для проверки
    print(f"Level {level} filter coefficients:")
    print("H0 coefficients:", H0.values)
    print("F0 coefficients:", F0.values)
    print("H1 coefficients:", H1.values)
    print("F1 coefficients:", F1.values)

    # Проведение анализа
    y0, y1 = signal.Analysis(H0, H1)

    # Сохранение результатов анализа
    analysis_output_path_0 = os.path.join(output_folder, f"analysis_output_0_{level}.wav")
    analysis_output_path_1 = os.path.join(output_folder, f"analysis_output_1_{level}.wav")
    list_to_wav(analysis_output_path_0, y0.values, sample_rate)
    list_to_wav(analysis_output_path_1, y1.values, sample_rate)

    # Сохранение аудиограммы для результата анализа
    plot_spectrogram(np.array(y0.values), sample_rate,
                     os.path.join(output_folder, f"analysis_output_spectrogram_0_{level}.png"))
    plot_spectrogram(np.array(y1.values), sample_rate,
                     os.path.join(output_folder, f"analysis_output_spectrogram_1_{level}.png"))

    # Проведение синтеза
    reconstructed_signal = signal.Synthesis(y0, y1, F0, F1)

    # Сохранение результирующего аудиофайла
    synthesis_output_path = os.path.join(output_folder, f"synthesis_output_{level}.wav")
    list_to_wav(synthesis_output_path, reconstructed_signal.values, sample_rate)

    # Сохранение аудиограммы для синтезированного сигнала
    plot_spectrogram(np.array(reconstructed_signal.values), sample_rate,
                     os.path.join(output_folder, f"synthesis_output_spectrogram_{level}.png"))

    print(f"Результат сохранен в {synthesis_output_path}")

# Добавляем фильтры для анализа и синтеза
sq = math.sqrt(2)
h0 = Signal([1/sq, 1/sq], start_index=0)  # фильтр для анализа
h1 = Signal([1/sq, -1/sq], start_index=0)  # другой фильтр для анализа
f0 = Signal([1/sq, 1/sq], start_index=-1)  # фильтр для синтеза
f1 = Signal([-1/sq, 1/sq], start_index=-1)  # другой фильтр для синтеза

# Вызов функции с вашим аудиофайлом и путями для сохранения
audio_path = "Exp.mp3"
output_folder = "output_level_results"  # Папка для сохранения результатов
level = 6  # Вы можете изменять уровень

transform_audio_with_levels_and_save(audio_path, output_folder, level)