import numpy as np
from math import sqrt


def cal_error(target, prediction):
    error = target - prediction
    lens = len(target)

    squaredError = np.power(error, 2)
    absError = np.abs(error)
    absPercentError = np.divide(absError, target, out=np.zeros_like(absError, dtype=np.float64), where=target != 0)

    targetMean = np.sum(target) / lens  # target平均值
    targetDeviation = np.apply_along_axis(lambda x: (x - targetMean) ** 2, 0, target)

    ret = (sum(squaredError) / lens,  # MSE
           sqrt(sum(squaredError) / lens),  # RMSE
           sum(absError) / lens,  # MAE
           sum(absPercentError) * 100 / lens,  # MAPE
           sum(targetDeviation) / lens,  # variance
           sqrt(sum(targetDeviation) / lens)  # std
           )

    return map(lambda x: round(x, 4), ret)


if __name__ == "__main__":
    N = int(1e5)

    target = np.random.rand(N) * 2 - 1
    # target = np.random.randn(N)
    prediction = np.ones(N) * np.mean(target)
    MSE, RMSE, MAE, MAPE, var, std = cal_error(target, prediction)
    print(f'MSE: {MSE}, RMSE: {RMSE}, MAE: {MAE}, MAPE:{MAPE}%, var: {var}, std: {std}')

