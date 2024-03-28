import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt


class LineFit:

    def __init__(
        self,
        ms_list: list[str],
        chain: list[list[int]],
        invoke_times: list[list[int]],
        pod_num: list[int],
        graph_ratio: dict[list[float]],
    ):
        self.fit_a = []
        self.fit_b = []
        self.ml_a = []
        self.ml_b = []
        self.lamd1 = []
        self.lamd2 = []
        self.skew = []

        self.pod_num = pod_num
        self.ms_list = ms_list
        self.graph_ratio = [graph_ratio] * len(self.ms_list)
        self.chain = chain
        self.ms = np.arange(12).tolist()
        self.invoke_times = invoke_times

    def line_f(self, x, a, b):
        return a * x + b

    def line_params(self, service):
        data = pd.read_csv(f"data/{service}.csv")
        x = np.array(data["workloadPerPod"])
        fit_y = np.array(data["fit_pars"])
        ml_y = np.array(data["ml_pars"])
        fit_params, _ = curve_fit(f=self.line_f, xdata=x, ydata=fit_y, maxfev=500000)
        ml_params, _ = curve_fit(f=self.line_f, xdata=x, ydata=ml_y, maxfev=500000)
        self.fit_a.append(fit_params[0])
        self.fit_b.append(fit_params[1])
        self.ml_a.append(ml_params[0])
        self.ml_b.append(ml_params[1])
        skew = np.array(data["skew"])
        self.skew.append(round(skew.mean(), 3))

    def get_params(self):
        for ms in self.ms_list:
            self.line_params(ms)

        df = pd.DataFrame(
            [
                self.chain,
                self.ms,
                self.invoke_times,
                self.graph_ratio,
                self.fit_a,
                self.fit_b,
                self.ml_a,
                self.ml_b,
                self.skew,
                self.pod_num,
            ]
        )
        df.columns = self.ms_list
        df.index = [
            "chain",
            "ms",
            "n",
            "percentage",
            "a1",
            "b1",
            "a2",
            "b2",
            "s",
            "pod_num",
        ]

        data = []
        for ms in self.ms_list:
            ms_data = df[ms].tolist()
            if len(ms_data[0]) > 1:
                simple_ms_data = ms_data.copy()
                simple_ms_data[0] = ms_data[0][0]
                simple_ms_data[2] = ms_data[2][0]
                simple_ms_data[3] = ms_data[3][0]
                complex_ms_data = ms_data.copy()
                complex_ms_data[0] = ms_data[0][1]
                complex_ms_data[2] = ms_data[2][1]
                complex_ms_data[3] = ms_data[3][1]
                data.extend([simple_ms_data, complex_ms_data])
            else:
                chain = ms_data[0][0] - 1
                ms_data_copy = ms_data.copy()
                ms_data_copy[0] = ms_data[0][0]
                ms_data_copy[2] = ms_data[2][chain]
                ms_data_copy[3] = ms_data[3][chain]
                data.append(ms_data_copy)
        datas = pd.DataFrame(
            columns=[
                "chain",
                "ms",
                "n",
                "percentage",
                "a1",
                "b1",
                "a2",
                "b2",
                "s",
                "pod_num",
            ],
            data=data,
        )
        self.fit_a = []
        self.fit_b = []
        self.ml_a = []
        self.ml_b = []
        self.skew = []
        return datas


ms_list = [
    "ts-basic-service",
    "ts-seat-service",
    "ts-ticketinfo-service",
    "ts-travel2-service",
    "ts-travel-service",
    "ts-config-service",
    "ts-order-service",
    "ts-order-other-service",
    "ts-price-service",
    "ts-route-service",
    "ts-station-service",
    "ts-train-service",
]
pod_num = [18, 8, 15, 15, 8, 2, 6, 8, 1, 15, 15, 8]
graph_ratio = [0.9, 0.1]
chain = [
    [1,2],
    [1,2],
    [1,2],
    [1],
    [2],
    [1,2],
    [2],
    [1],
    [1,2],
    [1,2],
    [1,2],
    [1,2],
]
invoke_times = [
    [22, 100],
    [10, 60],
    [16, 64],
    [26, 0],
    [0, 113],
    [2, 12],
    [0, 18],
    [3, 0],
    [1, 6],
    [8, 26],
    [11, 50],
    [4, 24],
]

line_fit = LineFit(ms_list, chain, invoke_times, pod_num, graph_ratio)
data = line_fit.get_params()
data.to_csv("data/line_fit.csv", index=False)
