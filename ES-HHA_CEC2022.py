from copy import deepcopy
import os

from scipy.stats import levy
from opfunu.cec_based.cec2022 import *
from scipy.stats import pearsonr


DimSize = 10
PopSize = 100
Func_num = 1
Pop = np.zeros((PopSize, DimSize))
FitPop = np.zeros(PopSize)

MaxFEs = 1000 * DimSize
curIter = 0
MaxIter = int(MaxFEs / PopSize)
LB = [-100] * DimSize
UB = [100] * DimSize
R = 1
F = 0.8
alpha = 0.5
Trials = 30
Xbest = np.zeros(DimSize)
FitBest = None

def InitialPop(func):
    global PopSize, DimSize, Pop, FitPop, Xbest, FitBest
    Pop = np.zeros((PopSize, DimSize))
    FitPop = np.zeros(PopSize)
    for i in range(PopSize):
        for j in range(DimSize):
            Pop[i][j] = LB[j] + np.random.rand() * (UB[j] - LB[j])
        FitPop[i] = func.evaluate(Pop[i])
    best_idx = np.argmin(FitPop)
    FitBest = FitPop[best_idx]
    Xbest = deepcopy(Pop[best_idx])


def CheckIndi(Indi):
    global DimSize
    for i in range(DimSize):
        range_width = UB[i] - LB[i]
        if Indi[i] > UB[i]:
            n = int((Indi[i] - UB[i]) / range_width)
            mR = (Indi[i] - UB[i]) - (n * range_width)
            Indi[i] = UB[i] - mR
        elif Indi[i] < LB[i]:
            n = int((LB[i] - Indi[i]) / range_width)
            mR = (LB[i] - Indi[i]) - (n * range_width)
            Indi[i] = LB[i] + mR
        else:
            pass


def UniformSearch(Off, i):
    global DimSize, Pop, FitPop, Xbest, R
    for j in range(DimSize):
        Off[i][j] = Xbest[j] + np.random.uniform(-R, R)
    CheckIndi(Off[i])


def NormalSearch(Off, i):
    global DimSize, Pop, FitPop, Xbest, R
    for j in range(DimSize):
        Off[i][j] = Xbest[j] + R * np.random.normal(0, 1)
    CheckIndi(Off[i])


def LevySearch(Off, i):
    global DimSize, Pop, FitPop, Xbest
    for j in range(DimSize):
        Off[i][j] = Xbest[j] + levy.rvs()
    CheckIndi(Off[i])


def DE_best(Off, i):
    global DimSize, Pop, FitPop, Xbest, PopSize
    r1, r2 = np.random.choice(list(range(PopSize)), 2, replace=False)
    Off[i] = Xbest + F * (Pop[r1] - Pop[r2])
    CheckIndi(Off[i])


def DE_rand(Off, i):
    global DimSize, Pop, FitPop, PopSize
    r1, r2, r3 = np.random.choice(list(range(PopSize)), 3, replace=False)
    Off[i] = Pop[r1] + F * (Pop[r1] - Pop[r2])
    CheckIndi(Off[i])


def DE_cur(Off, i):
    global DimSize, Pop, FitPop, Xbest, PopSize
    candi = list(range(PopSize))
    candi.remove(i)
    r1, r2 = np.random.choice(candi, 2, replace=False)
    Off[i] = Pop[i] + F * (Pop[r1] - Pop[r2])
    CheckIndi(Off[i])


def DE_cur2best(Off, i):
    global DimSize, Pop, FitPop, Xbest, PopSize
    candi = list(range(PopSize))
    candi.remove(i)
    r1, r2 = np.random.choice(candi, 2, replace=False)
    Off[i] = Pop[i] + F * (Xbest - Pop[i]) + F * (Pop[r1] - Pop[r2])
    CheckIndi(Off[i])


def DE_cur2pbest(Off, i):
    global DimSize, Pop, FitPop, PopSize
    sort_idx = np.argsort(FitPop)[0:int(0.05 * PopSize)]
    Xmean = np.mean(Pop[sort_idx], axis=0)
    candi = list(range(PopSize))
    candi.remove(i)
    r1, r2 = np.random.choice(candi, 2, replace=False)
    Off[i] = Pop[i] + F * (Xmean - Pop[i]) + F * (Pop[r1] - Pop[r2])
    CheckIndi(Off[i])


def Diversity(Pop):
    global LB, UB
    X_mean = np.mean(Pop, axis=0)
    diverse = 0
    for indi in Pop:
        sumup = 0
        for i in range(len(indi)):
            sumup += abs(indi[i] - X_mean[i]) / (UB[i] - LB[i])
        diverse += sumup / len(indi)
    return diverse / len(Pop)


def Distance(X, Y):
    dis = 0
    for i in range(len(X)):
        dis += (X[i] - Y[i]) ** 2
    return np.sqrt(dis)


def FDC(Pop, Fit):
    size = len(Pop)
    X_best = Pop[np.argmin(Fit)]
    dis = np.zeros(size)
    for i in range(size):
        dis[i] = Distance(Pop[i], X_best)
    return pearsonr(dis, Fit)[0]


def Metric(Pop, Fit):
    global alpha
    metric = alpha * Diversity(Pop) + (1 - alpha) * FDC(Pop, Fit)
    if metric > 0.9:
        metric = 0.9
    elif metric < 0.1:
        metric = 0.1
    return metric


def ESHHA(func):
    global PopSize, DimSize, curIter, MaxIter, Pop, FitPop, Xbest, FitBest

    InitialPop(func)
    Off = np.zeros((PopSize, DimSize))
    FitOff = np.zeros(PopSize)

    Exploration = [DE_rand, DE_cur, DE_cur2best, DE_cur2pbest]
    Exploitation = [UniformSearch, NormalSearch, LevySearch, DE_best]
    Trace = []
    for i in range(MaxIter):

        metric = Metric(Pop, FitPop)
        for j in range(PopSize):
            if np.random.rand() < metric:
                Search = np.random.choice(Exploitation)
            else:
                Search = np.random.choice(Exploration)
            Search(Off, j)
            FitOff[j] = func.evaluate(Off[j])
            if FitOff[j] < FitPop[j]:
                FitPop[j] = FitOff[j]
                Pop[j] = deepcopy(Off[j])
                if FitOff[j] < FitBest:
                    Xbest = deepcopy(Off[j])
                    FitBest = FitOff[j]
        Trace.append(FitBest)
    return Trace


def main(dim):
    global DimSize, LB, UB, MaxFEs, MaxIter, Trials
    DimSize = dim
    LB = [-100] * dim
    UB = [100] * dim

    PopSize = 100
    MaxFEs = 1000 * dim
    MaxIter = int(MaxFEs / PopSize)

    CEC2022Funcs = [F12022(dim), F22022(dim), F32022(dim), F42022(dim), F52022(dim), F62022(dim), F72022(dim),
                    F82022(dim), F92022(dim), F102022(dim), F112022(dim), F122022(dim)]

    for i in range(len(CEC2022Funcs)):
        All_Trial_Best = []
        for j in range(Trials):
            np.random.seed(2022 + 88 * j)
            Trace = ESHHA(CEC2022Funcs[i])
            All_Trial_Best.append(Trace)
        np.savetxt("./ES-HHA_Data/CEC2022/F" + str(i + 1) + "_" + str(dim) + "D.csv", All_Trial_Best, delimiter=",")


if __name__ == "__main__":
    if os.path.exists('ES-HHA_Data/CEC2022') == False:
        os.makedirs('ES-HHA_Data/CEC2022')
    Dims = [10, 20]
    for dim in Dims:
        main(dim)

