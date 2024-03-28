import numpy as np


class end2endLatencyPrediction:
    def __init__(self):

        self.chain = 2

    def latency2probability(self, sla):

        self.df["sTemp"] = self.df.s * self.df.n
        probability = 0
        for chain in range(self.chain):
            parameterTemp = 1
            df = self.df[self.df.chain == chain + 1]
            tempConv = 0
            for i, row_i in df.iterrows():
                temp = row_i.lambdaValue
                for j, row_j in df.iterrows():
                    if row_i.ms != row_j.ms:
                        temp = temp * (row_j.lambdaValue - row_i.lambdaValue)
                tempConv += (
                    row_i.percentage * (1 - np.exp(-row_i.lambdaValue * sla)) / temp
                )
                parameterTemp = parameterTemp * np.exp(row_i.sTemp * row_i.lambdaValue)
            probability += np.prod(df["lambdaValue"]) * tempConv
        return probability

    def monotonous(self):
        for sla in range(100):
            print("sla:{}, probability:{}.".format(sla, self.latency2probability(sla)))

    def sla_for_p95(self, temp):
        self.df = temp.copy()
        slaMin = 1
        slaMax = 10000
        sla = (slaMin + slaMax) / 2
        epsilon = 0.0001
        flag = True
        iteration = 0
        percentageTarget = 0.95

        while flag:
            iteration += 1
            temp = self.latency2probability(sla)
            if temp > percentageTarget - epsilon and temp < percentageTarget + epsilon:
                break
            elif temp < percentageTarget - epsilon:
                slaMin = sla
                sla = (slaMin + slaMax) / 2
            else:
                slaMax = sla
                sla = (slaMin + slaMax) / 2
            if iteration > 20:
                flag = False
        return (sla, flag)
