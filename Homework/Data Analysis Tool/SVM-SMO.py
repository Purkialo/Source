# -*-coding:utf-8 -*-
import numpy as np
import random

"""
Author:
	Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Zhihu:
    https://www.zhihu.com/people/Jack--Cui/
Modify:
	2017-10-03
"""

class optStruct:
	"""
	数据结构，维护所有需要操作的值
	Parameters：
		dataMatIn - 数据矩阵 
		classLabels - 数据标签
		C - 松弛变量
		toler - 容错率
	"""
	def __init__(self, dataMatIn, classLabels, C, toler):
		self.X = dataMatIn								#数据矩阵
		self.labelMat = classLabels						#数据标签
		self.C = C 										#松弛变量
		self.tol = toler 								#容错率
		self.m = np.shape(dataMatIn)[0] 				#数据矩阵行数
		self.alphas = np.mat(np.zeros((self.m,1))) 		#根据矩阵行数初始化alpha参数为0	
		self.b = 0 										#初始化b参数为0
		self.eCache = np.mat(np.zeros((self.m,2))) 		#根据矩阵行数初始化虎误差缓存，第一列为是否有效的标志位，第二列为实际的误差E的值。

def loadDataSet(fileName):
	"""
	读取数据
	Parameters:
	    fileName - 文件名
	Returns:
	    dataMat - 数据矩阵
	    labelMat - 数据标签x
	"""
	dataMat = []; labelMat = []
	fr = open(fileName)
	for line in fr.readlines():                                     #逐行读取，滤除空格等
		lineArr = line.strip().split(',')
		dataMat.append(lineArr[:-1])      					  #添加数据
		labelMat.append(lineArr[-1])                          #添加标签
	
	return dataMat,labelMat

def preprocessor(dataMat,labelMat):
	nums2lables = {}
	lables2nums = {}
	classnum = 1
	for i in labelMat:
		if not(i in nums2lables.values()):
			nums2lables[classnum] = i
			lables2nums[i] = classnum
			classnum += 1
	for i in range(len(labelMat)):
		labelMat[i] = lables2nums[labelMat[i]]
	data = np.concatenate((dataMat,np.array([labelMat]).T),axis = 1)
	delnum = []
	for i in range(len(data)):
		if(data[i][-2] == '?'):
			delnum.append(i)
		if(data[i][-1] == '2'):
			data[i][-1] = -1
	data = np.delete(data,delnum,axis = 0)
	for i in range(7):
		data[:,i] = translable(data[:,i])
	data = data.astype(int)
	return data[:,:-1],data[:,-1]

def translable(labelMat):
	lables2nums = {}
	classnum = 1
	for i in labelMat:
		if not(i in lables2nums.keys()):
			lables2nums[i] = classnum
			classnum += 1
	for i in range(len(labelMat)):
		labelMat[i] = lables2nums[labelMat[i]]
	return labelMat

def calcEk(oS, k):
	"""
	计算误差
	Parameters：
		oS - 数据结构
		k - 标号为k的数据
	Returns:
	    Ek - 标号为k的数据误差
	"""
	fXk = float(np.multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T) + oS.b)#n*1矩阵
	Ek = fXk - float(oS.labelMat[k])
	return Ek

def selectJrand(i, m):
	"""
	函数说明:随机选择alpha_j的索引值

	Parameters:
	    i - alpha_i的索引值
	    m - alpha参数个数
	Returns:
	    j - alpha_j的索引值
	"""
	j = i                                 #选择一个不等于i的j
	while (j == i):
		j = int(random.uniform(0, m))
	return j

def selectJ(i, oS, Ei):
	"""
	内循环启发方式2
	Parameters：
		i - 标号为i的数据的索引值
		oS - 数据结构
		Ei - 标号为i的数据误差
	Returns:
	    j, maxK - 标号为j或maxK的数据的索引值
	    Ej - 标号为j的数据误差
	"""
	maxK = -1; maxDeltaE = 0; Ej = 0 						#初始化
	oS.eCache[i] = [1,Ei]  									#根据Ei更新误差缓存
	validEcacheList = np.nonzero(oS.eCache[:,0].A)[0]		#返回误差不为0的数据的索引值
	if (len(validEcacheList)) > 1:							#有不为0的误差
		for k in validEcacheList:   						#遍历,找到最大的Ek
			if k == i: continue 							#不计算i,浪费时间
			Ek = calcEk(oS, k)								#计算Ek
			deltaE = abs(Ei - Ek)							#计算|Ei-Ek|
			if (deltaE > maxDeltaE):						#找到maxDeltaE
				maxK = k; maxDeltaE = deltaE; Ej = Ek
		return maxK, Ej										#返回maxK,Ej
	else:   												#没有不为0的误差
		j = selectJrand(i, oS.m)							#随机选择alpha_j的索引值
		Ej = calcEk(oS, j)									#计算Ej
	return j, Ej 											#j,Ej

def updateEk(oS, k):
	"""
	计算Ek,并更新误差缓存
	Parameters：
		oS - 数据结构
		k - 标号为k的数据的索引值
	Returns:
		无
	"""
	Ek = calcEk(oS, k)										#计算Ek
	oS.eCache[k] = [1,Ek]									#更新误差缓存

def clipAlpha(aj,H,L):
	"""
	修剪alpha_j
	Parameters:
	    aj - alpha_j的值
	    H - alpha上限
	    L - alpha下限
	Returns:
	    aj - 修剪后的alpah_j的值
	"""
	if aj > H: 
		aj = H
	if L > aj:
		aj = L
	return aj

def innerL(i, oS):
	"""
	优化的SMO算法
	Parameters：
		i - 标号为i的数据的索引值
		oS - 数据结构
	Returns:
		1 - 有任意一对alpha值发生变化
		0 - 没有任意一对alpha值发生变化或变化太小
	"""
	#步骤1：计算误差Ei
	Ei = calcEk(oS, i)
	#优化alpha,设定一定的容错率。
	if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
		#使用内循环启发方式2选择alpha_j,并计算Ej
		j,Ej = selectJ(i, oS, Ei)
		#保存更新前的aplpha值，使用深拷贝
		alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy()
		#步骤2：计算上下界L和H
		if (oS.labelMat[i] != oS.labelMat[j]):
			L = max(0, oS.alphas[j] - oS.alphas[i])
			H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
		else:
			L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
			H = min(oS.C, oS.alphas[j] + oS.alphas[i])
		if L == H: 
			print("L==H")
			return 0
		#步骤3：计算eta
		eta = 2.0 * oS.X[i,:] * oS.X[j,:].T - oS.X[i,:] * oS.X[i,:].T - oS.X[j,:] * oS.X[j,:].T
		if eta >= 0: 
			print("eta>=0")
			return 0
		#步骤4：更新alpha_j
		oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej)/eta
		#步骤5：修剪alpha_j
		oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
		#更新Ej至误差缓存
		updateEk(oS, j)
		if (abs(oS.alphas[j] - alphaJold) < 0.00001): 
			print("alpha_j变化太小")
			return 0
		#步骤6：更新alpha_i
		oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])
		#更新Ei至误差缓存
		updateEk(oS, i)
		#步骤7：更新b_1和b_2
		b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[i,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
		b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[j,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[j,:].T
		#步骤8：根据b_1和b_2更新b
		if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
		elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
		else: oS.b = (b1 + b2)/2.0
		return 1
	else: 
		return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter):
	'''
	完整的线性SMO算法
	Parameters：
		dataMatIn - 数据矩阵
		classLabels - 数据标签
		C - 松弛变量
		toler - 容错率
		maxIter - 最大迭代次数
	Returns:
		oS.b - SMO算法计算的b
		oS.alphas - SMO算法计算的alphas
	'''
	oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler)					#初始化数据结构
	innerL(1,oS)
	iter = 0 																						#初始化当前迭代次数
	entireSet = True; alphaPairsChanged = 0
	while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):							#遍历整个数据集都alpha也没有更新或者超过最大迭代次数,则退出循环
		alphaPairsChanged = 0
		if entireSet:																				#遍历整个数据集   						
			for i in range(oS.m):        
				alphaPairsChanged += innerL(i,oS)													#使用优化的SMO算法
				print("全样本遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter,i,alphaPairsChanged))
			iter += 1
		else: 																						#遍历非边界值
			nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]						#遍历不在边界0和C的alpha
			for i in nonBoundIs:
				alphaPairsChanged += innerL(i,oS)
				print("非边界遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter,i,alphaPairsChanged))
			iter += 1
		if entireSet:																				#遍历一次后改为非边界遍历
			entireSet = False
		elif (alphaPairsChanged == 0):																#如果alpha没有更新,计算全样本遍历 
			entireSet = True  
		print("迭代次数: %d" % iter)
	return oS.b,oS.alphas 																			#返回SMO算法计算的b和alphas

def calcWs(alphas,dataArr,classLabels):
	"""
	计算w
	Parameters:
		dataArr - 数据矩阵
	    classLabels - 数据标签
	    alphas - alphas值
	Returns:
	    w - 计算得到的w
	"""
	X = np.mat(dataArr)
	labelMat = np.mat(classLabels).T
	m,n = np.shape(X)
	w = np.zeros((n,1))
	for i in range(m):
		w += np.multiply(alphas[i]*labelMat[i],X[i,:].T)
	return w

def classifer(data,w,b):
	res = np.mat([data]) * np.mat(w) + b
	return int(np.sign(res))

if __name__ == '__main__':
	dataArr, classLabels = loadDataSet('post-operative.data')
	dataArr,classLabels = preprocessor(dataArr,classLabels)
	#print(dataArr,classLabels)
	lentrain = int(len(dataArr) * 0.80)
	lentest = len(dataArr) - lentrain
	#print(dataArr)
	#print(classLabels)
	b, alphas = smoP(dataArr[:], classLabels[:], 0.6, 0.001, 40)
	w = calcWs(alphas,dataArr[:], classLabels[:])
	#print(w)
	#print(b)
	flag_w = 0
	for i in range(lentest):
		resclass = classifer(dataArr[lentrain + i],w,b)
		target = classLabels[lentrain + i]
		if(resclass != target):
			flag_w += 1
			print("Predicted class: %s, actually: %s, it's wrong!" % (str(resclass),str(target)))
		else:
			print("Predicted class: %s, actually: %s, it's right!" % (str(resclass),str(target)))
	print("Accuracy: ",(lentest - flag_w)/lentest)
