# coding=utf-8

# 主程序需要用到的库
from scipy.fftpack import fft, ifft
import math
from scipy import signal, stats
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose, STL, DecomposeResult
import pandas as pd
import matplotlib.pyplot as plt
import sys
import time

# 主程序正在尝试使用深度学习方案，这些是用到的库
import tensorflow.compat.v1 as tf # 教程为tensorflow v1版本，兼容v1函数的workaround

# 测试需要用到的库
import json

class TimeSeriesPredictor(object):
    """根据给定的时间序列预测未来结果

    支持预测曲线的绘制
    支持预测结果的验证

    属性:
        一、在构造函数中需要/可选指定的属性:

        timeSeries: pandas.Series对象，numpy.ndarray对象或者列表
            预测所依据的时间序列。
        predictNum: (可选)整数
            默认为给定时间序列长度

        二、预测/用户设置相关属性：

        1)预测设置相关属性
        说明：
            所有预测设置相关属性可由setPredictOptions()函数设置
            这些属性会被预测方法和检验方法使用，并影响预测准确度。
        __decomposeFunction: 函数对象
            时间序列分解函数。可选，默认为stl分解
        __trendPredictFunction: 函数对象
            趋势预测函数。可选，默认为多项式拟合预测
        __trendPredictDegree: int
            拟合多项式的最高次数。可选，默认为3

        2)用户设置相关属性
        说明：
            所有用户设置相关属性都可由setUserOptions()函数设置
            这些属性调整某些用户功能的行为
        __drawingsAutoDisplay: 布尔型
            按照默认值为True时，调用绘制函数之后立即显示图形窗口；
            或者可以设置为False，并且在在调用所有绘制函数之后使用displayAllDrawings()函数一次性绘制所有图像
        __saveDrawings: 布尔型
            按照默认值为False，各个绘制的图像不会被保存为文件
        __printResult: 布尔型
            默认值为False。当指定为True时，计算某些数值的同时会打印出结果

        三、其他属性:

        1)计算步骤的结果
        period: 浮点数/整数
            解析得到的时间序列的周期。由__extractPeriod()方法得到，并由__roundThePeriod()方法取整
        decomposeResult: statsmodels.tsa.seasonal.DecomposeResult实例
            时间序列分解的结果。包含各个分解部分并且支持直接使用plot()方法绘制分解结果
        trendPredictResult: pandas.Series实例
            趋势部分预测的结果。
        seasonalPredictResult: pandas.Series实例
            季节性部分预测的结果。
        predictResult: pandas.Series实例
            预测结果

        2)用于绘制图像的属性
        periodExtractDetails: 字典
            解析时间序列过程中的细节信息，用于绘制周期分析图像

        3)验证过程/结果
        validate_predictor: TimeSeriesPredictor实例
            保存了验证预测过程/结果的TimeSeriesPredictor实例
        validate_actualSeries: numpy.ndarray对象
            验证预测部分对应的真实值。用于验证的准确度计算/验证图像的绘制。
        validate_ape: numpy.ndarray对象
            用于衡量真实值和预测值之间的误差，各点的APE值(绝对百分比误差)。用于MAPE的计算以及APE图像的绘制
        validate_mape: 浮点数
            用于衡量真实值和预测值的误差的MAPE值(平均百分比误差)

    基本步骤
        1. 实例化TimeSeriesPredictor
        2. (可选)使用setPredictOptions()设置预测选项/使用setUserOptions()设置用户选项
        3. 调用predict()方法进行预测
        4. (可选)运用绘制函数分析预测结果以及预测过程
        5. (可选)运用validate_predict()方法进行内部交叉验证，以估计结果的准确度

    """

    ## 内置函数
    def __init__(self, timeSeries, predictNum=0):
        self.timeSeries = timeSeries
        self.__convertTimeSeriesToNumpyNdarray()
        if predictNum <= 0:
            predictNum = int(len(self.timeSeries))
        self.predictNum = predictNum
        self.setPredictOptions()
        self.setUserOptions()

    def __str__(self):
        """for debug"""
        displayStr = ""
        displayStr += "-" * 8 + "\n"
        displayStr += "Object: TimeSeriesPredictor\n"
        displayStr += "len of timeSeries: {}\n".format(len(self.timeSeries))
        displayStr += "-" * 8 + "\n"
        return displayStr

    ## 预测
    # 预测函数
    # def predict(self):
    #     """
    #     预测时间序列的未来值。函数返回预测结果
    #     所有预测过程/结果会被保存在实例中
    #     """
    #     # 提取周期
    #     self.__extractPeriod()
    #     # 分解时间序列
    #     self.decomposeResult = self.__decomposeFunction(self.timeSeries, self.period)  # type: DecomposeResult
    #     # 趋势性预测
    #     self.trendPredictResult = self.__trendPredictFunction(self.decomposeResult.trend, self.predictNum,
    #                                                           self.__trendPredictDegree)
    #     # 周期性预测
    #     self.__seasonalPredict()
    #     # 整合各部分预测结果
    #     self.__composeTrendAndSeasonal()
    #     return self.predictResult.copy()

    # 另一种预测：深度学习预测方案
    def predict(self, learning_rate=0.1, maxIteration=10000, targetError=0.01):
        # def predict_ANNs(self):
        sampleSize = len(self.timeSeries)
        sampleTimeMat = np.array([range(sampleSize)]).transpose()  # type: np.ndarray
        print("sampleTime:", sampleTimeMat)
        sampleValueMat = np.array([self.timeSeries]).transpose()  # type: np.ndarray
        # 注意：这里无视类属性self.predictNum预测长度，预测长度必须与样本长度一致，否则提取结果时矩阵行数不匹配
        predictTimeMat = np.array([range(sampleSize, sampleSize+self.predictNum)]).transpose() # type: np.ndarray
        print("predictTime:", predictTimeMat)
        predictValueMat = np.array([]) # type: np.ndarray
        # 归一化
        sampleTimeMat = (sampleTimeMat - sampleTimeMat.mean()) / (
                    sampleTimeMat.max() - sampleTimeMat.min())
        sampleValueMat_mean = sampleValueMat.mean()
        sampleValueMat_range = sampleValueMat.max() - sampleValueMat.min()
        sampleValueMat = (sampleValueMat - sampleValueMat_mean) / sampleValueMat_range
        predictTimeMat = (predictTimeMat - predictTimeMat.mean()) / (
                    predictTimeMat.max() - predictTimeMat.min())
        # sampleTimeMat = sampleTimeMat / (
        #         sampleTimeMat.max() - sampleTimeMat.min())
        # sampleValueMat_range = sampleValueMat.max() - sampleValueMat.min()
        # sampleValueMat = sampleValueMat / sampleValueMat_range
        # predictTimeMat = predictTimeMat / (
        #         predictTimeMat.max() - predictTimeMat.min())
        tf.compat.v1.disable_eager_execution()  # 解决placeholder和eager execution不兼容问题
        x = tf.placeholder(tf.float32, [None, 1])
        y = tf.placeholder(tf.float32, [None, 1])
        # 隐藏层
        w1 = tf.Variable(tf.random_uniform([1, 10], 0, 1))
        b1 = tf.Variable(tf.zeros([1, 10]))
        wb1 = tf.matmul(x, w1) + b1
        layer1 = tf.nn.relu(wb1)
        # 输出层
        w2 = tf.Variable(tf.random_uniform([10, 1], 0, 1))
        b2 = tf.Variable(tf.zeros([sampleSize, 1]))
        wb2 = tf.matmul(layer1, w2) + b2
        # layer2 = tf.nn.relu(wb2)
        layer2 = wb2
        # 梯度下降
        loss = tf.reduce_mean(tf.square(y - layer2)) # 方差
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            iteration = 1
            error = 1
            progress = 0
            # while any([iteration <= maxIteration, error > targetError]):
            while iteration <= maxIteration:
                session.run(train_step, feed_dict={x:sampleTimeMat, y:sampleValueMat})
                # error = tf.sqrt(session.run(loss, feed_dict={x:sampleTimeMat, y:sampleValueMat})) # 由于归一化处理，误差为介于[0, 1]之间的小数
                currentProgress = round(iteration / maxIteration * 100, 2)
                if currentProgress != progress:
                    progress = currentProgress
                    self.__printInfo("Progress: {:.2f}%".format(progress))
                iteration += 1
            print("iteration:", iteration, "\nerror:", error)
            predictValueMat = session.run(layer2, feed_dict={x:predictTimeMat})
            predictValueMat = predictValueMat * sampleValueMat_range + sampleValueMat_mean
            # predictValueMat = predictValueMat * sampleValueMat_range
        self.predictResult = predictValueMat.transpose()[0]
        return self.predictResult.copy()

    ## 内部计算过程
    # 周期提取函数
    def __extractPeriod(self, samplingFrequency=1):
        """
        根据时间序列求取周期

        :param samplingFrequency: int
            用于进行周期单位换算的可选参数，默认为1
        :return: none
        """
        ## 第一部分：FFT处理时间序列，计算幅谱图中的幅值和频率
        N = len(self.timeSeries)  # 采样点数
        amplitudes = abs(fft(self.timeSeries))
        # 双边幅谱取一半处理
        amplitudes = amplitudes[0: math.ceil(N / 2)]
        # 计算幅值
        for i, amplitude in enumerate(amplitudes):
            if i == 0:
                amplitudes[i] = amplitudes[i] / N
            else:
                amplitudes[i] = amplitudes[i] * 2 / N
        # 计算频率
        frequencies = [i * samplingFrequency / N for i in range(len(amplitudes))]
        ## 第二部分：获取幅值最大的正弦波的周期作为结果
        # 获取幅值极值
        decomposed_indexes = signal.argrelextrema(amplitudes, np.greater)
        # 按照幅值降序排序
        decomposed_amplitudes = amplitudes[decomposed_indexes]
        sorted_indexes = np.argsort(-decomposed_amplitudes)
        # 筛选、排序后的频率
        result_frequencies = np.array(frequencies)[decomposed_indexes][sorted_indexes]

        # 周期过滤(workaround)
        # 注：针对于部分序列会得到与样本数相等的周期的情况(例如：y = x ** 2)做的workaround：如果当前周期大于样本数的1/2，此周期对预测无意义，抛弃此周期
        periodFilter = result_frequencies > 2 / N
        result_frequencies = result_frequencies[periodFilter]
        # 选取合适频率，并根据频率计算周期
        if len(result_frequencies) == 0:
            self.period = 2
            self.__printInfo("No period found. Set period = 2.")
        elif result_frequencies[0] != 0:
            self.period = 1 / result_frequencies[0]
        else:
            self.period = 1 / result_frequencies[1]
        # 周期取整
        # 注：时间序列分解函数需要给定的周期是整数。不是整数的周期会被四舍五入处理，并且因为会影响测试结果而给出警告
        if self.period - math.floor(self.period) != 0:
            self.__printWarning(
                "Period '{}' is not an integer and has been rounded. Predictions may be inaccurate.".format(
                    self.period))
        self.period = int(round(self.period))
        # 用于绘制图像
        self.periodExtractDetails = {}
        self.periodExtractDetails["frequencies"] = frequencies
        self.periodExtractDetails["amplitudes"] = amplitudes
        self.periodExtractDetails["resultFrequencies"] = result_frequencies
        self.periodExtractDetails["resultAmplitudes"] = np.array(decomposed_amplitudes)[sorted_indexes][periodFilter]

    # 季节性预测函数
    def __seasonalPredict(self):
        # 根据往期值预测未来一周期的值
        futureCycle = pd.Series([], dtype='float64')  # type: pd.Series
        for i in range(self.period):
            # 获取所有往期对应值，求众数，作为周期中该时刻预测依据
            pastValues = self.decomposeResult.seasonal[-1 - i:0:-self.period]
            mode = stats.mode(pastValues)[0][0]
            futureCycle = futureCycle.append(pd.Series(mode, dtype='float64'), ignore_index=True)
        # print(futureCycle)
        # 对未来一周期的值进行重复，并补齐余数部分，获得完整的季节性预测结果
        futureSeasonal = pd.Series(futureCycle.to_list() * (self.predictNum // self.period), name="seasonal",
                                   dtype='float64')
        futureSeasonal = futureSeasonal.append(futureCycle[0: self.predictNum % self.period], ignore_index=True)
        self.seasonalPredictResult = futureSeasonal

    # 整合各部分预测结果
    def __composeTrendAndSeasonal(self):
        self.predictResult = self.trendPredictResult + self.seasonalPredictResult  # type: pd.Series

    ## 外部计算过程
    # 时间序列分解函数
    # 注：各个分解函数需要具有相同的参数和返回值
    @classmethod
    def classicalDecomposition(cls, timeSeries, period):
        """
        经典分解法。根据指定的周期，用经典分解法分解给定的时间序列

        :param timeSeries: pandas.Series对象或numpy.ndarray对象
            时间序列
            注意：只能是pandas.Series对象或numpy.ndarray对象，因为statsmodels.tsa.seasonal.STL不支持时间序列是列表
        :param period: 整数
            时间序列的周期。
            须为整数
        :return: statsmodels.tsa.seasonal.DecomposeResult
        """
        return seasonal_decompose(timeSeries, period=period)  # type: DecomposeResult

    @classmethod
    def stlDecomposition(cls, timeSeries, period):
        """
        STL分解法。根据指定的周期，用STL分解法分解给定的时间序列

        :param timeSeries:
        :param period:
        :return:
        """
        return STL(timeSeries, period=period, robust=True).fit()  # type: DecomposeResult

    # 趋势预测函数
    # 注：各个趋势预测函数需要具有相同的参数和返回值
    @classmethod
    def polyFitTrendPredict(cls, knownTrend, predictNum, degree=3):
        """
        多项式拟合预测

        :param existingTrend: pandas.Series，numpy.ndarray或者列表
            时间序列分解得到的趋势部分
        :param predictNum: 整数
            需要预测多少个未来值
        :return: pandas.Series
            这个分解部分的预测结果。未来趋势/周期性的时间序列
        """
        x = np.array(range(len(knownTrend)))
        y = knownTrend
        params = np.polyfit(x, y, degree)
        params_func = np.poly1d(params)
        x_predict = np.array(range(len(knownTrend), len(knownTrend) + predictNum))
        y_predict = pd.Series([params_func(i) for i in x_predict])
        return y_predict

    ## 作图
    # 显示所有已经绘制的图像
    def displayAllDrawings(self):
        plt.show()

    # 被绘制函数调用，按照对应属性控制是否立即显示图像
    def __displayCurrentDrawing(self):
        if self.__drawingsAutoDisplay:
            plt.show()

    def __saveDrawing(self, text):
        if self.__saveDrawings:
            plt.savefig("{}_{}.png".format(self.__class__.__name__, text))

    # 周期提取图
    def drawPeriodExtractDetails(self):
        plt.figure()
        plt.title("PeriodExtractDetail (Result: {})".format(self.period))
        inputCurve = plt.plot(self.timeSeries)[0]
        frequenciesCurve = plt.plot(self.periodExtractDetails["frequencies"], color="red")[0]
        amplitudesCurve = plt.plot(self.periodExtractDetails["amplitudes"], color="orange")[0]
        plt.legend(handles=[inputCurve, frequenciesCurve, amplitudesCurve],
                   labels=["input", "frequencies", "amplitudes"])
        self.__saveDrawing("PeriodExtractDetail")
        self.__displayCurrentDrawing()

    # 时间序列分解图
    def drawDecomposeResult(self):
        self.decomposeResult.plot()
        plt.subplot(4, 1, 1)
        plt.title("DecomposeResult")  # 覆盖原有第一个子视图的标题"value"
        self.__saveDrawing("DecomposeResult")
        self.__displayCurrentDrawing()

    # 绘制已知曲线和预测曲线
    def __drawKnownAndFutureValues(self, title, knownValues, futureValues):
        plt.figure()
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Value")
        knownLen = len(knownValues)
        # 这两处转换为numpy ndarray是为了避免numpy future warning告警
        if not isinstance(knownValues, np.ndarray):
            knownValues = knownValues.to_numpy()
        if not isinstance(futureValues, np.ndarray):
            futureValues = futureValues.to_numpy()
        plt.plot(range(knownLen), knownValues)
        plt.plot(range(knownLen, knownLen + len(futureValues)), futureValues, "--")
        self.__saveDrawing(title)
        self.__displayCurrentDrawing()

    # 趋势部分预测图
    def drawTrendPredictResult(self):
        self.__drawKnownAndFutureValues("TrendPredictResult", self.decomposeResult.trend, self.trendPredictResult)

    # 季节性部分预测图
    def drawSeasonalPredictResult(self):
        self.__drawKnownAndFutureValues("SeasonalPredictResult", self.decomposeResult.seasonal,
                                        self.seasonalPredictResult)

    # 最终预测图
    def drawPredictResult(self):
        self.__drawKnownAndFutureValues("PredictResult", self.timeSeries, self.predictResult)

    ## 检验
    # 交叉验证的预测步骤
    def validate_predict(self, trainingPercent=0.7, *args, **kwargs):
        # 划分样本集和测试集
        totalNum = len(self.timeSeries)
        trainingNum = round(totalNum * trainingPercent)
        trainingSeries = self.timeSeries[0:trainingNum]
        actualSeries = self.timeSeries[trainingNum:]
        # 对样本进行验证
        validatePredictor = TimeSeriesPredictor(trainingSeries, totalNum - trainingNum)
        validatePredictor.setPredictOptions(**self.getPredictOptions())
        validatePredictor.setUserOptions(**self.getUserOptions())
        validateResult = validatePredictor.predict(*args, **kwargs)
        # 数据保存
        self.validate_predictor = validatePredictor
        self.validate_actualSeries = actualSeries
        return validateResult.copy()

    # 检查是否执行过验证预测函数
    def __validatePredictPerformed(self):
        # 检查是否已经进行验证性预测，且预测结果有效
        examineList = ["validate_predictor", "validate_actualSeries"]
        return all([hasattr(self, i) for i in examineList])

    # 计算各点APE(绝对百分比误差)
    def validate_calculateApe(self):
        if not self.__validatePredictPerformed():
            self.__printWarning(
                "Need to perform function '{}' before performing '{}'.".format(self.validate_predict.__name__,
                                                                               self.validate_calculateApe().__name__))
            return
        if 0 in self.validate_actualSeries:
            self.__printWarning("One or more actual values equal 0. APE can't be calculated.")
            return
        ape = (self.validate_predictor.predictResult - self.validate_actualSeries) / self.validate_actualSeries * 100
        if self.__printResult:
            self.__printDebug("APE of the validation result: {}".format(ape))
        self.validate_ape = ape
        return ape

    # 计算MAPE(平均绝对百分比误差)
    def validate_calculateMape(self):
        if not self.__validatePredictPerformed():
            self.__printWarning(
                "Need to perform function '{}' before performing '{}'.".format(self.validate_predict.__name__,
                                                                               self.validate_calculateMape.__name__))
            return
        if 0 in self.validate_actualSeries:
            self.__printWarning("One or more actual values equal 0. MAPE can't be calculated.")
            return
        self.validate_calculateApe()
        mape = np.mean(np.abs(self.validate_ape))
        if self.__printResult:
            self.__printDebug("MAPE of the validation result: {:.2f}%".format(mape))
        self.validate_mape = mape
        return mape

    # 绘制交叉验证图像
    def validate_drawValidationResult(self):
        if not self.__validatePredictPerformed():
            self.__printWarning(
                "Need to perform function '{}' before performing '{}'.".format(self.validate_predict.__name__,
                                                                               self.validate_drawValidationResult.__name__))
            return
        title = "ValidationResult"
        plt.figure()
        if not hasattr(self, "validate_mape"):
            self.validate_calculateMape()
        plt.title("{} (MAPE: {:.2f}%)".format(title, self.validate_mape))
        plt.xlabel("Time")
        plt.ylabel("Value")
        timeSeries = self.timeSeries
        predictSeries = self.validate_predictor.predictResult
        if not isinstance(timeSeries, np.ndarray):
            timeSeries = timeSeries.to_numpy()
        if not isinstance(predictSeries, np.ndarray):
            predictSeries = predictSeries.to_numpy()
        trainingLen = len(self.validate_predictor.timeSeries)
        totalLen = len(self.timeSeries)
        plt.plot(range(totalLen), timeSeries)
        plt.plot(range(trainingLen, totalLen), predictSeries, "--")
        self.__saveDrawing(title)
        self.__displayCurrentDrawing()

    # 绘制误差分布图像
    def validate_drawErrorDistribution(self, foucusOnPrediction=True):
        """
        绘制各预测点的误差分布情况(APE值)

        :param foucusOnPrediction: 布尔型
            按照默认值true只会展示预测部分的误差分布。如果指定为False则会同时显示出完整的原始图像
        :return:
        """
        if not self.__validatePredictPerformed():
            self.__printWarning(
                "Need to perform function '{}' before performing '{}'.".format(self.validate_predict.__name__,
                                                                               self.validate_drawErrorDistribution.__name__))
            return
        title = "ErrorDistribution"
        if not hasattr(self, "validate_mape"):
            self.validate_calculateMape()
        timeSeries = self.timeSeries
        predictSeries = self.validate_predictor.predictResult
        if not isinstance(timeSeries, np.ndarray):
            timeSeries = timeSeries.to_numpy()
        if not isinstance(predictSeries, np.ndarray):
            predictSeries = predictSeries.to_numpy()
        trainingLen = len(self.validate_predictor.timeSeries)
        totalLen = len(self.timeSeries)
        plt.subplots(2, 1)
        # 子图1：已知曲线和预测曲线对比
        plt.subplot(2, 1, 1)
        plt.title("{} (MAPE: {:.2f}%)".format(title, self.validate_mape))
        plt.ylabel("Value")
        if foucusOnPrediction:
            plt.plot(range(trainingLen, totalLen), timeSeries[trainingLen:])
        else:
            plt.plot(range(totalLen), timeSeries)
        plt.plot(range(trainingLen, totalLen), predictSeries, "--")
        xlim = plt.xlim()
        # 子图2：各点APE值
        plt.subplot(2, 1, 2)
        plt.xlabel("Time")
        plt.ylabel("APE (%)")
        plt.xlim(xlim)
        plt.bar(range(trainingLen, totalLen), self.validate_ape)
        self.__saveDrawing(title)
        self.__displayCurrentDrawing()

    ## 预测/用户选项
    # 设置预测选项
    def setPredictOptions(self, decomposeFunction=None, trendPredictFunction=None, trendPredictDegree=3):
        if not decomposeFunction:
            self.__decomposeFunction = self.stlDecomposition
        if not trendPredictFunction:
            self.__trendPredictFunction = self.polyFitTrendPredict
        self.__trendPredictDegree = trendPredictDegree

    # 获取预测选项
    def getPredictOptions(self):
        predictOptionsDict = {}
        predictOptionsDict["decomposeFunction"] = self.__decomposeFunction
        predictOptionsDict["trendPredictFunction"] = self.__trendPredictFunction
        predictOptionsDict["trendPredictDegree"] = self.__trendPredictDegree
        return predictOptionsDict

    # 设置用户选项
    def setUserOptions(self, drawingsAutoDisplay=True, saveDrawings=False, printResult=False):
        self.__drawingsAutoDisplay = drawingsAutoDisplay
        self.__saveDrawings = saveDrawings
        self.__printResult = printResult

    # 获取用户选项
    def getUserOptions(self):
        userOptionsDict = {}
        userOptionsDict["drawingsAutoDisplay"] = self.__drawingsAutoDisplay
        userOptionsDict["saveDrawings"] = self.__saveDrawings
        userOptionsDict["printResult"] = self.__printResult
        return userOptionsDict

    ## 内部工具
    # 时间序列转换为numpy.ndarray对象
    def __convertTimeSeriesToNumpyNdarray(self):
        """
        将时间序列转换为numpy.ndarray对象支持部分处理函数
        注：不能使用列表，STL函数不支持列表
        注：不能直接转换为pandas.Series对象，因为会导致周期解析函数不支持

        :return:
        """
        if isinstance(self.timeSeries, pd.Series) or isinstance(self.timeSeries, np.ndarray):
            return
        elif isinstance(self.timeSeries, list):
            self.timeSeries = np.array(self.timeSeries, dtype=np.float64)
            return
        else:
            self.__printWarning(
                "The type of timeSeries is not numpy.ndarray, pandas.Series or list. Some functions might not support!")

    # 打印警告
    def __printWarning(self, warningInfo):
        """
        打印警告

        为所打印的内容加上类名和Warning前缀

        :param warningInfo: 字符串
            所打印的警告提示信息

        :return:
        """
        print("[{}] WARNING: {}".format(self.__class__.__name__, warningInfo))

    # 打印INFO信息
    def __printInfo(self, info):
        print("[{}] INFO: {}".format(self.__class__.__name__, info))

    # 打印DEBUG信息
    def __printDebug(self, debugInfo):
        print("[{}] DEBUG:".format(self.__class__.__name__))
        print("\t<Time: {}>".format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
        print("\t<Code: Line {} in function '{}', called by function '{}'>".format(sys._getframe(1).f_lineno,
                                                                                   sys._getframe(1).f_code.co_name,
                                                                                   sys._getframe(2).f_code.co_name))
        print("{}".format(debugInfo))


## ---- Get the input ----
# input = pd.read_excel(r"testData1 - seasonal.xlsx")["value"] # 注意：这里需要转化为一维数组然后传进FFT函数，否则会出现问题
# input = pd.read_excel(r"testData2 - seasonal with trend.xlsx")["value"]
# input = pd.read_excel(r"testData3 - seasonal with trend, with abnormal values.xlsx")["value"]
# input = [math.sin(x) for x in range(100)]
# input = [1 + 5 * math.sin(2 * math.pi * 200 * x) + 7 * math.sin(2 * math.pi * 400 * x) + 3 * math.sin(2 * np.pi * 600 * x) for x in np.linspace(0, 1, num=1400)]
# input = [math.sin(2 * math.pi * 0.25 * x) for x in range(100)] # 周期为4的正弦曲线
# input = [math.sin(x) + x for x in range(100)]
# 完美世界(002624)2021数据
with open("cn_002624_2021.json", "rt") as f:
    historyQueryList = json.load(f)[0]["hq"]
input = [i[2] for i in historyQueryList]
# input = [x ** 2 + 200 * math.sin(2 * math.pi * 0.25 * x) for x in range(200)]
# input = [x ** 2 for x in range(200)]
# input = [x ** 2 + 2 for x in range(200)]
# input = [4 * x + 3 + 2 * math.sin(2 * math.pi * 0.25 * x) for x in range(100)]
# input = [0 for i in range(200)]

# debug
# input = pd.read_excel(r"debugData1 - trend.xlsx")["value"]
# input = [i for i in range(6)]
# timeSeriesPredictor1 = TimeSeriesPredictor(input, 8)
# input = [i for i in range(1000)]
# input = [i for i in range(1000)] + [1000 - i for i in range(1000)]

## ---- Generate the output ----

# timeSeriesPredictor1 = TimeSeriesPredictor(input)
# timeSeriesPredictor1 = TimeSeriesPredictor(input, len(input))
# timeSeriesPredictor1.predict(decomposeFunction=TimeSeriesPredictor.classicalDecomposition)
# timeSeriesPredictor1.predict(decomposeFunction=TimeSeriesPredictor.stlDecomposition)
# timeSeriesPredictor1.setPredictOptions(trendPredictDegree=2)
# timeSeriesPredictor1.setUserOptions(drawingsAutoDisplay=False, saveDrawings=True, printResult=False)
# timeSeriesPredictor1.predict()
# timeSeriesPredictor1.validate_predict()


## ---- Debug ----
# print(timeSeriesPredictor1.period)
# TimeSeriesPredictor.classicalDecomposition(timeSeriesPredictor1.timeSeries, timeSeriesPredictor1.period).plot()
# TimeSeriesPredictor.classicalDecomposition(timeSeriesPredictor1.timeSeries, 3.5).plot() # 时间序列分解函数不支持整数
# print(TimeSeriesPredictor.stlDecomposition(timeSeriesPredictor1.timeSeries, timeSeriesPredictor1.period))
# plt.show()
# timeSeriesPredictor1.drawDecomposeResult()
# print(timeSeriesPredictor1.periodExtractDetails)
# print(TimeSeriesPredictor.smaTrendPredict(timeSeriesPredictor1.decomposeResult.trend, 10))
# print(timeSeriesPredictor1.decomposeResult.seasonal)
# print(timeSeriesPredictor1.seasonalPredictResult)
# print(timeSeriesPredictor1.period)
# print(timeSeriesPredictor1.predict())
# print(timeSeriesPredictor1.predictNum)
# print(timeSeriesPredictor1.timeSeries)
# print(len(timeSeriesPredictor1.timeSeries))
# print("input", timeSeriesPredictor1.decomposeResult.trend)
# print("trendPredict", timeSeriesPredictor1.trendPredictResult)
# print(TimeSeriesPredictor.smaTrendPredict([i for i in range(6)], 6, 3))
# print(timeSeriesPredictor1.getUserOptions())
# print(timeSeriesPredictor1.validate_predictor.getUserOptions())

# input = input[:int(len(input)/2)]
if len(input) % 2 == 1:
    input = input[:-1]
timeSeriesPredictor1 = TimeSeriesPredictor(input)
timeSeriesPredictor1.setUserOptions(drawingsAutoDisplay=False, saveDrawings=True, printResult=False)
timeSeriesPredictor1.validate_predict(trainingPercent=0.5, maxIteration=10000)

# 周期提取相关
# print(timeSeriesPredictor1.validate_predictor.periodExtractDetails)

## ---- Print ----
# timeSeriesPredictor1.validate_calculateApe()
# timeSeriesPredictor1.validate_calculateMape()

## ---- Display ----
# timeSeriesPredictor1.drawPeriodExtractDetails()
# timeSeriesPredictor1.drawDecomposeResult()
# timeSeriesPredictor1.drawTrendPredictResult()
# timeSeriesPredictor1.drawSeasonalPredictResult()
# timeSeriesPredictor1.drawPredictResult()

timeSeriesPredictor1.validate_drawValidationResult()
timeSeriesPredictor1.validate_drawErrorDistribution(foucusOnPrediction=True)
# timeSeriesPredictor1.validate_predictor.drawPeriodExtractDetails()
# timeSeriesPredictor1.validate_predictor.drawDecomposeResult()
# timeSeriesPredictor1.validate_predictor.drawTrendPredictResult()
# timeSeriesPredictor1.validate_predictor.drawSeasonalPredictResult()
# timeSeriesPredictor1.validate_predictor.drawPredictResult()
timeSeriesPredictor1.displayAllDrawings()
