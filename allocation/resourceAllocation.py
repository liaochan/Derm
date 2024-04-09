import numpy as np
import pandas as pd
import ipdb
import math
from end2endLatencyUpdate import end2endLatencyPrediction


class OptimizationProblem:
    def __init__(self, sla, workload, ratio):
        # profiling result
        self.df = pd.read_csv("data/all_pars.csv")
        self.ratio = ratio
        self.df["percentage"][self.df.percentage == 0.9] = self.ratio
        self.df["percentage"][self.df.percentage == 0.1] = 1 - self.ratio
        self.df["gap"] = 0
        self.profilingWorkload = 40
        self.chain = 2
        self.iterations = 30
        self.epsilon = 0
        self.sla = sla
        self.workload = workload
        self.cutoffLambda = 0.8
        self.miniWorkload = 10
        self.epsilonLambda = 0.00001
        # number of allocated containers must less than 100000000
        self.allocated_resource_min = 100000000
        self.pai = 0
        self.rauMultiplyAcc = [0, 0, 0]
        self.dataProcessing()
        self.initOpt()

    def dataProcessing(self):
        # transform from one MS to n MS
        self.df["a"] = self.df["a1"] / self.df.n
        self.df["b"] = self.df["b1"] / self.df.n

        # the range of MS w
        self.df["cutoffMax"] = (self.cutoffLambda - self.df["b1"]) / self.df["a1"]
        self.df["cutoffMin"] = (
            self.miniWorkload
            * self.df["percentage"]
            * self.df["n"]
            / self.df["pod_num"]
        )
        tempDF = self.df.groupby("ms")["cutoffMin"].sum().reset_index()
        self.df = pd.merge(self.df.drop(columns="cutoffMin"), tempDF, on="ms")
        self.df["wTuning"] = (self.df["cutoffMin"] + self.df["cutoffMax"]) / 2

        # compute the initial s_0 for all ms
        self.df["sTemp"] = self.df["s"] * self.df.n
        self.df["sExp"] = np.exp(self.df["sTemp"])

        self.df["msWorkload"] = (
            self.df["n"] * self.df["percentage"] * self.profilingWorkload
        )
        msWorkloadTemp = self.df.groupby(["ms"]).msWorkload.sum().reset_index()
        msPodTemp = self.df.groupby(["ms"]).pod_num.mean().reset_index()
        mergeTemp = pd.merge(msWorkloadTemp, msPodTemp, on="ms")
        mergeTemp["w"] = mergeTemp.msWorkload / mergeTemp.pod_num
        self.df = pd.merge(self.df, mergeTemp, on="ms")

    def initOpt(self):
        # consistent to the profiling model
        self.df["rau"] = 0
        self.df["pai"] = 1
        # initialize w, lambda, pai and rau
        self.df["lambdaValue"] = self.df["w"] * self.df["a"] + self.df["b"]
        self.w_order(1)
        self.df["lambdaMultiply"] = 0
        self.lambda_multiply()
        self.pai_update()
        self.rau_update()

    def lambda_multiply(self):
        # return the result of lambda_1 * lambda_3 * lambda_5 ...
        for chain in range(self.chain):
            chain += 1
            dfTemp = self.df[self.df.chain == chain]
            temp = 1
            for i in range(len(dfTemp)):
                i += 1
                if i % 2 == 1:
                    row = dfTemp[dfTemp.order == i].iloc[0]
                    if i == len(dfTemp):
                        temp = temp * row.lambdaValue
                    else:
                        temp = temp * np.power(row.lambdaValue, 2)
            self.df.lambdaMultiply[self.df.chain == chain] = temp

    def w_order(self, iteration):
        if iteration == 0:
            self.df["sensitivity"] = self.df["a"] * self.df["percentage"]
            dfTempList = []
            for chain in range(self.chain):
                chain += 1
                dfTemp = self.df[self.df.chain == chain]
                sensitivity = (
                    dfTemp[["chain", "ms", "sensitivity"]]
                    .sort_values("sensitivity")
                    .reset_index(drop=True)
                    .reset_index()
                    .drop(columns="sensitivity")
                )
                # use the index to decide the equal lambda
                dfTempList.append(pd.merge(dfTemp, sensitivity, on=["chain", "ms"]))
            if self.chain > 1:
                self.df = pd.concat([dfTempList[0], dfTempList[1]])
            else:
                self.df = dfTempList[0]
            # lambda_1 = lambda_2 < lambda_3 = lambda_4
            self.df["order"] = self.df["index"] + 1
            self.df = self.df.drop(columns="index")
        else:
            # reorder to keep lambda_3 > lambda_1, lambda_5 > lambda_3, ...
            self.df["sensitivity"] = self.df["a"] * self.df["percentage"]
            dfTempList = []
            for chain in range(self.chain):
                chain += 1
                dfTemp = self.df[self.df.chain == chain]
                sensitivity = (
                    dfTemp[["chain", "ms", "lambdaValue"]]
                    .sort_values("lambdaValue")
                    .reset_index(drop=True)
                    .reset_index()
                    .drop(columns="lambdaValue")
                )
                # use the index to decide the equal lambda
                dfTempList.append(pd.merge(dfTemp, sensitivity, on=["chain", "ms"]))
            if self.chain > 1:
                self.df = pd.concat([dfTempList[0], dfTempList[1]])
            else:
                self.df = dfTempList[0]
            # lambda_1 = lambda_2 < lambda_3 = lambda_4
            self.df["order"] = self.df["index"] + 1
            self.df = self.df.drop(columns="index")

    def rau_multiply_acc_update(self):
        for chain in range(self.chain):
            chain += 1
            dfTemp = self.df[(self.df.chain == chain) & self.df.order % 2 == 1]
            self.rauMultiplyAcc[chain] = np.prod(dfTemp["rau"])

    def w_merge_and_update(self):
        # mean normalize different w of the same MS
        self.df["w"] = (self.df.lambdaValue - self.df.b) / self.df.a
        tempMean = self.df.groupby(["ms"]).w.mean().reset_index()
        temp = self.df.drop(columns="w")
        self.df = pd.merge(temp, tempMean, on="ms")

        # achieve the proper range
        self.df.w[self.df.w < self.df.cutoffMin] = self.df.cutoffMin
        self.df.w[self.df.w > self.df.cutoffMax] = self.df.cutoffMax

        # update the lambda of normalized MS
        self.df["lambdaValue"] = self.df.w * self.df.a + self.df.b

        # compute the allocated resource
        self.df["resource"] = (
            self.df["percentage"] * self.df["n"] * self.workload / self.df["w"]
        )

    def rau_update(self):
        def coefficient_first(i, lambda_i, dfTemp):
            if len(dfTemp) == 2:
                return 1
            elif i == len(dfTemp):
                return 0
            else:
                temp = 1
                for index in range(len(dfTemp)):
                    index += 1
                    if index % 2 == 1 and index != i:
                        row = dfTemp[dfTemp.order == index].iloc[0]
                        # the ms with the max(lamdba_i)
                        if index == len(dfTemp):
                            temp = temp / (row.lambdaValue - lambda_i)
                        else:
                            temp = temp / np.power(row.lambdaValue - lambda_i, 2)
                return temp

        def coefficient_second(i, lambda_i, coeFirst, dfTemp):
            if len(dfTemp) == 2:
                return 0
            else:
                if coeFirst == 0:
                    temp = 1
                    for index in range(len(dfTemp)):
                        index += 1
                        if index % 2 == 1 and index != i:
                            row = dfTemp[dfTemp.order == index].iloc[0]
                            temp = temp / np.power(row.lambdaValue - lambda_i, 2)
                else:
                    temp = 0
                    for index in range(len(dfTemp)):
                        index += 1
                        if index % 2 == 1 and index != i:
                            row = dfTemp[dfTemp.order == index].iloc[0]
                            if index == len(dfTemp):
                                temp += -coeFirst / (row.lambdaValue - lambda_i)
                            else:
                                temp += -2 * coeFirst / (row.lambdaValue - lambda_i)
                return temp

        # only compute rau_1, rau_3, rau_5, .... for each chain
        for chain in range(self.chain):
            chain += 1
            dfTemp = self.df[self.df.chain == chain]
            for i in range(len(dfTemp)):
                i += 1
                if i % 2 == 1:
                    try:
                        row = dfTemp[dfTemp.order == i].iloc[0]
                    except:
                        ipdb.set_trace()
                    lambda_i = row.lambdaValue
                    coeFirst = coefficient_first(i, lambda_i, dfTemp)
                    coeSecond = coefficient_second(i, lambda_i, coeFirst, dfTemp)
                    rauTemp = (
                        coeFirst * (lambda_i * self.sla + 1) / (lambda_i * lambda_i)
                        + coeSecond / lambda_i
                    )
                    if rauTemp <= 0:
                        self.df.rau[(self.df.chain == chain) & (self.df.order == i)] = (
                            self.df.rau.mean()
                        )
                    else:
                        self.df.rau[(self.df.chain == chain) & (self.df.order == i)] = (
                            rauTemp * row["lambdaMultiply"]
                        )

    def lambda_update(self, iteration):
        for chain in range(self.chain):
            chain += 1
            chainLenght = len(self.df[self.df.chain == chain])
            for i in range(chainLenght):
                dfTemp = self.df[self.df.chain == chain]
                i += 1

                # MS No. is even
                if i % 2 == 0:
                    pairRow = dfTemp[dfTemp.order == i - 1].iloc[0]
                    # lambda_2 = lambda_1, lambda_4 = lambda_3, ...
                    self.df.gap[(self.df.chain == chain) & (self.df.order == i)] = (
                        self.df[
                            (self.df.chain == chain) & (self.df.order == i)
                        ].lambdaValue
                        - pairRow["lambdaValue"]
                    )
                    self.df.loc[
                        (self.df.chain == chain) & (self.df.order == i), "lambdaValue"
                    ] = (pairRow["lambdaValue"] + self.epsilonLambda)

                # update the last MS if the number of MS is odd since the update of this MS is different to others.
                elif i == len(dfTemp):
                    row = dfTemp[dfTemp.order == i].iloc[0]
                    temp_i = (
                        -row.percentage
                        * self.workload
                        * row["n"]
                        / (row.w * row.w * row.a)
                    )
                    # temp = - np.log(temp_i / (self.pai * row.rau * self.sla * row.sExp)) / self.sla
                    temp = -np.log(temp_i / (self.pai * row.rau * self.sla)) / self.sla
                    self.df.gap[(self.df.chain == chain) & (self.df.order == i)] = (
                        temp - row["lambdaValue"]
                    )
                    self.df.loc[
                        (self.df.chain == chain) & (self.df.order == i), "lambdaValue"
                    ] = temp

                #  MS No. is odd
                else:
                    row = dfTemp[dfTemp.order == i].iloc[0]
                    pairRow = dfTemp[dfTemp.order == i + 1].iloc[0]
                    temp_i = (
                        -row.percentage
                        * self.workload
                        * row["n"]
                        / (row.w * row.w * row.a)
                    )
                    temp_W = (
                        -pairRow.percentage
                        * self.workload
                        * pairRow["n"]
                        / (pairRow.w * pairRow.w * pairRow.a)
                    )
                    # temp = - np.log((temp_i + temp_W) / (self.pai * row.rau * self.sla * row.sExp)) / self.sla
                    temp = (
                        -np.log((temp_i + temp_W) / (self.pai * row.rau * self.sla))
                        / self.sla
                    )
                    self.df.gap[(self.df.chain == chain) & (self.df.order == i)] = (
                        temp - row["lambdaValue"]
                    )
                    self.df.loc[
                        (self.df.chain == chain) & (self.df.order == i), "lambdaValue"
                    ] = temp

    def pai_update(self):
        temp = 0
        for chain in range(self.chain):
            chain += 1
            dfTemp = self.df[self.df.chain == chain]
            for i in range(len(dfTemp)):
                i += 1
                row = dfTemp[dfTemp.order == i].iloc[0]
                if i == len(dfTemp):
                    # the ms with the max(order)
                    tempTemp = (
                        self.workload
                        * row["n"]
                        * row.percentage
                        / (row.w * row.w * row.a)
                        / self.sla
                    )
                    temp += tempTemp
                elif i % 2 == 1:
                    # parameters of ms_2, ms_4, ms_6, ....
                    pairRow = dfTemp[dfTemp.order == i + 1].iloc[0]
                    #  a pair of equal lambda, we only compuate one of them.
                    #  For example, we choose the ms with index as 1, 3, 5, ...
                    tempTemp = (
                        self.workload
                        * (
                            row["n"] * row.percentage / (row.w * row.w * row.a)
                            + pairRow["n"]
                            * pairRow.percentage
                            / (pairRow.w * pairRow.w * pairRow.a)
                        )
                        / self.sla
                    )
                    temp += tempTemp
        self.pai = -20 * temp

    def iteractive_update(self):
        latencyPrediction = end2endLatencyPrediction()
        # update order for each ms: (rau, w) -> pai -> lambda -> (w, rau)
        for i in range(self.iterations):
            self.w_merge_and_update()
            self.rau_update()
            self.rau_multiply_acc_update()

            # latencyFlag is False if the latencyPrediction can not find out a proper SLA to satisfied the P95 within a given number of iteration.
            latency_value, latency_flag = latencyPrediction.sla_for_p95(self.df)
            allocated_resource = self.df["resource"].sum()

            # update the optimized allocation if the allocated resource is less than that of the previous iteration
            if (
                latency_flag
                and latency_value <= self.sla
                and allocated_resource < self.allocated_resource_min
            ):
                self.allocated_resource_min = allocated_resource
                temp_w = self.df.groupby("ms")["w"].mean().reset_index()
                temp_resource = self.df.groupby("ms")["resource"].sum().reset_index()
                self.df_parameter = pd.merge(temp_w, temp_resource, on="ms")

            self.lambda_update(i)
            self.lambda_multiply()
            self.pai_update()
            self.w_order(i + 1)
        try:
            return self.df_parameter["resource"].apply(lambda x: math.ceil(x))
        except:
            return 0


sla_list = [550]
workload_list = pd.read_csv("data/workload_ceil.csv")["workload"].tolist()
alloc = []
counter = 0
for i in workload_list:
    counter += 1
    for j in sla_list:
        print(f"Workload: {i}, SLA: {j}")
        opt = OptimizationProblem(sla=j, workload=i, ratio=0.5)
        res = opt.iteractive_update()
        alloc.append(res)
alloc = np.array(alloc)
alloc.sort()
print(alloc)
# Allocation results will be used for deployment...
