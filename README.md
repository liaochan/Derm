# Artifact Evaluation of Derm
We present Derm, a SLA-aware Resource Management for Highly Dynamic Microservices. To underscore the efficacy of Derm, we have made our artifact available on Zenodo, comprising the complete source code for our proposed implementation. This encompasses all components, such as the scheduler, and functionalities discussed in the paper, including the latency profiler, dynamic predictor, and allocation optimizer. Additionally, users have access to example training and profiling scripts, as well as a sample dataset from the TrainTicket benchmark, which can be effortlessly customized to suit their individual models and datasets. Example configuration and explanation can be found in our codes.

Due to time constraints, we have only uploaded the core codes for profiling, training, and allocation optimization. The complete version, including detailed evaluation, will be published in the near future. Stay tuned for updates on our GitHub repository.

## Experiment Environment
Machine Specifications:
| Resource Type | Value |
| :------------ | ----: |
| # of Nodes | 10 |
| Node CPU | 52 cores |
| Node RAM | 128GB |
| Container CPU | 0.2 cores |
| Container RAM | 1GB |

Software / Dependencies:
| Name | Version |
| :------- | ------: |
| Kubernetes | 1.23.2 |
| Python | 3.11.4 |
| matplotlib | 3.6.0 |
| pandas | 1.5.0 |
| numpy | 1.23.4 |
| seaborn | 0.12.1 |
| scikit-learn | 1.2.0 |
| xgboost | 1.7.2 |
| torch | 1.13.1 |

You can install all python dependencies by running the following command:
```bash
pip install -r requirements.txt
```

## Content

The folder structure is organized as follow:
```
profiling
├─data // Sample data that we used for profiling.
├─curve_cdf.py // Fit distribution of microservice latency.
├─line_fit_exp.py // Fit corelation of \lambda and \omega.
└─predict.py // Predict dynamic graph.

allocation
├─data // Our profiling output.
├─end2endLatencyUpdate.py // Used by resourceAllocation.py.
└─resourceAllocation.py // Core of resource allocation algorithms.
```

Below we also provide the sample code usage:

Profiling & Traning.
```bash
cd profiling
python curve_cdf.py
python -W ignore line_fit_exp.py
```

Graph Prediction
```bash
cd profiling
# Training
python predict.py train
# Testing
python predict.py test
```

Compute Allocation Scheme
```bash
cd allocation
python -W ignore resourceAllocation.py
```
