# coding=utf-8

from scipy.fftpack import fft, ifft
import math
from scipy import signal
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

# Amplitude: 幅值
def extractPeriod(timeSeries, samplingFrequency=1):
    """
    :param timeSeries: 一维数组，例如列表，或者一维np.array
    :param samplingFrequency: 取样频率，用于预测取1就可以。在默认情况下这个值为1，此函数计算出的周期单位为时间序列的最小间隔(比如：天)；当指定这个参数时，周期的单位会被自动转换为国际单位秒
    :return: 周期值
    """
    ## 第一部分：FFT处理时间序列，计算幅谱图中的幅值和频率
    N = len(timeSeries) # 采样点数
    amplitudes = abs(fft(timeSeries))
    # 双边幅谱取一半处理
    amplitudes = amplitudes[0: math.ceil(N/2)]
    # 计算幅值
    for i, amplitude in enumerate(amplitudes):
        if i == 0:
            amplitudes[i] = amplitudes[i] / N
        else:
            amplitudes[i] = amplitudes[i] * 2 / N
    # 计算频率
    frequencies = [i * samplingFrequency / N for i in range(len(amplitudes))]
    # (debug)绘制图像
    plt.plot(timeSeries)
    plt.plot(frequencies, color="red")
    plt.plot(amplitudes, color="orange")
    # plt.show()
    ## 第二部分：获取幅值最大的正弦波的周期作为结果
    # 获取幅值极值
    decomposed_indexes = signal.argrelextrema(amplitudes, np.greater)
    # 按照幅值降序排序
    decomposed_amplitudes = amplitudes[decomposed_indexes]
    sorted_indexes = np.argsort(-decomposed_amplitudes)
    # 排序后的频率和幅值
    result_frequencies = np.array(frequencies)[decomposed_indexes][sorted_indexes]
    result_amplitudes = np.array(amplitudes)[decomposed_indexes][sorted_indexes]
    print("frequencies: ", result_frequencies)
    print("amplitudes: ", result_amplitudes)
    # 返回幅值最大的正弦波的周期
    if result_frequencies[0] != 0:
        return 1 / result_frequencies[0]
    return 1 / result_frequencies[1]

## ---- Get the input ----
# input = [math.sin(x) for x in range(100)]
# input = [1 + 5 * math.sin(2 * math.pi * 200 * x) + 7 * math.sin(2 * math.pi * 400 * x) + 3 * math.sin(2 * np.pi * 600 * x) for x in np.linspace(0, 1, num=1400)]
input = [1 + 5 * math.sin(2 * math.pi * 200 * x) + 7 * math.sin(2 * math.pi * 400 * x) + 3 * math.sin(2 * np.pi * 600 * x) for x in np.linspace(0, 1, num=4096)]

# input = pd.read_excel(r"testData1 - seasonality.xlsx")["value"] # 注意：这里需要转化为一维数组然后传进FFT函数，否则会出现问题
# input = pd.read_excel(r"testData2 - seasonality with trend.xlsx")["value"]
# input = pd.read_excel(r"testData3 - seasonality with trend, with abnormal values.xlsx")["value"]

# input = pd.read_excel(r"Monthly static.xlsx")["count"]
# input = pd.read_csv(r"commit_number.csv")["commmit_number_total"]
# input = pd.read_csv(r"commit_number.csv")["commmit_number_config"]
# input = pd.read_csv(r"commit_number.csv")["commmit_number_product"]

## ---- Generate the output ----
print("period:", extractPeriod(input))
plt.show()
