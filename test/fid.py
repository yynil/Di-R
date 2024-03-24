import numpy as np
import scipy.stats as stats


# x.shape [H, W]
# y.shape [H, W]
def calc_fid(x: np.ndarray, y: np.ndarray) -> float:
    x_mu, x_sigma = x.mean(axis=0), np.cov(x, rowvar=False)
    y_mu, y_sigma = y.mean(axis=0), np.cov(y, rowvar=False)
    x_mu, y_mu = np.atleast_1d(x_mu), np.atleast_1d(y_mu)
    x_sigma, y_sigma = np.atleast_2d(x_sigma), np.atleast_2d(y_sigma)
    diff = (x_mu - y_mu) ** 2
    cov_mean = x_sigma.dot(y_sigma)
    cov_mean[cov_mean < 0] = 0
    cov_mean = np.sqrt(cov_mean)
    fid = diff.sum() + np.trace(x_sigma + y_sigma - cov_mean * 2)
    return fid


# x.shape [H, W]
# y.shape [H, W]
def calc_is(x: np.ndarray, y: np.ndarray) -> float:
    x_mu, y_mu = x.mean(axis=0), y.mean(axis=0)
    x_kl, y_kl = stats.entropy(x.T, qk=x_mu), stats.entropy(y.T, qk=y_mu)
    is_val = np.exp(np.mean([x_kl, y_kl]))
    return is_val


# x.shape [B, H, W]
# y.shape [B, H, W]
def calc_fid_batch(x: np.ndarray, y: np.ndarray) -> float:
    res = 0.
    for i in range(x.shape[0]):
        res += calc_fid(x[i], y[i])
    return res / x.shape[0]


# x.shape [B, H, W]
# y.shape [B, H, W]
def calc_is_batch(x: np.ndarray, y: np.ndarray) -> float:
    res = 0.
    for i in range(x.shape[0]):
        res += calc_is(x[i], y[i])
    return res / x.shape[0]


def test_fid():
    x = np.random.randn(300, 300, 300)
    y = np.random.randn(300, 300, 300)
    x = (x - x.min()) / (x.max() - x.min()) * 256
    y = (y - y.min()) / (y.max() - y.min()) * 256
    res = calc_fid_batch(x, y)
    print(res)
    res = calc_is_batch(x, y)
    print(res)


if __name__ == "__main__":
    test_fid()