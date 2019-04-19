from numpy import *
import operator
from os import listdir

# 2.1.1 创建数据集和标签
def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.1], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# 2.1.2 KNN算法
def classify0(inX, dataSet, labels, k):
    """
    计算点间的距离，然后对数据按照从小到大的次序排序。
    确定前k个距离最小元素所在的主要分类，输入k总是正整数
    将classCount字典分解为元组列表，使用item（）方法，按照第二个元素的次序对元组进行排序（逆序=大到小）
    返回发生频率最高的元素标签

    :param inX: 用于分类的向量
    :param dataSet: 训练样本集
    :param labels: 标签向量（与dataSet的行数相同）
    :param k: 用于选择最近邻居的数目
    :return:
    """
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()  # Indicies 指数
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


#  2.2.1 准备数据：从文本文件中解析数据——文本记录转换为Numpy
def file2matrix(filename):
    '''
    处理输入格式问题
    打开文件，得到文件的行数
    创建以0填充的矩阵Numpy（实际为一个二维数组），将矩阵的另一维度设置为固定值3（简化处理，可更改）
    使用line.strip()截取掉所有的回车字符，使用tab（'\t'）将上一步得到的整行数据分割成一个元素列表。选取前3个元素，将其存储到特征矩阵中
    （python中-1表示列表中的最后一列元素——负索引）将列表的最后一列存储到向量classLabelVector中（必须告知解释器元素值为整型，否则python会将这些元素当作字符串）


    :param filename:文件名字符串
    :return: 训练样本矩阵和类标签向量
    '''
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()  # 去掉头尾的\n \t等空格符号
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


#  2.2.3 准备数据：归一化数值
def autoNorm(dataSet):
    '''
    newValue = (oldValue - min)/(max - min)
    每列的最小值在minValue中，最大值在maxValue中。dataSet.shape(0)中的参数0使得函数从列中选取最小值
    函数计算可能的取值范围，并创建新的返回矩阵
    根据公式归一化特征值。然而特征值矩阵有1000*3个值，而minValue和range的值都为1*3。故使用tile()函数将变量内容复制成输入矩阵同样大小的矩阵
    这是具体特征值相除。在Numpy库中，矩阵除法需要使用  linalg.solve(matA, matB)

    :param dataSet:
    :return:
    '''
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


#  2.2.4 测试算法：作为完整程序验证分类器
def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')  # 书中是datingTestSet.txt 这里是作者的错误，str没办法转成int型
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :],
                                     datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d"
              % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is : %f" % (errorCount / float(numTestVecs)))


#  使用算法。约会网站预测函数
def classfyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input(
        "percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classfierResults = classify0((inArr - minVals)
                                 / ranges, normMat, datingLabels, 3)
    print("You will probably like this person: ",
          resultList[classfierResults - 1])




# 识别数字系统

# 图像格式转换成分类器使用的向量格式
def img2vector(filename):
    '''
    创建1*1024的np数组，打开文件，循环读出文件前32行，并将每行的头32个字符值存储在np数组中

    :param filename:
    :return:  np数组
    '''
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect


# 测试算法：识别手写数字
def handwritingClassTest():
    '''
    将目录中的文件内容存储在列表中，得到目录中文件的数目，存储到m中
    创建一个m行1024列的训练矩阵，每行数据存储一个图像。可以从文件名中解析出分类数字
    该目录下的文件按照规则命名（9_45.txt的分类是9，数字9的第45个实例）。存储到hwLabels向量中
    用img2vector函数载入图像。
    使用classify0()函数测试该目录下的每个文件。

    改变变量k的值、修改函数handwritingClassTest随机选取训练样本、改变训练样本的数目，都会对KNN算法的错误率产生影响


    :return:
    '''
    hwLabels = []
    trainingFileList = listdir('trainingDigits')  #要先从os中导入listdir包
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector("trainingDigits//%s" % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector("testDigits/%s" % fileNameStr)
        classifierResult = classify0(vectorUnderTest,
                                     trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real anwser is: %d"
              % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount/float(mTest)))
