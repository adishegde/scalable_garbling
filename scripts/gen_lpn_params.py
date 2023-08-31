import argparse
from lpn_param import analysisforq
from scipy.stats import binom


def correctness_error(N, k, tau):
    d = N - k + 1
    max_weight = (d - 1) // 2
    return 1 - binom.cdf(max_weight, N, 1.0 / 2**tau)


def sec_lpn(N, k, tau, q):
    t = (N * (q - 1)) // (2**tau * q)
    try:
        return analysisforq(N, k, t, q)
    except:
        return 0


def find_k_correct(N, tau, error, k_beg=None, k_end=None):
    if k_beg is None:
        k_beg = 1
    if k_end is None:
        k_end = N

    if k_beg + 1 >= k_end:
        return k_beg

    k_mid = (k_beg + k_end) // 2
    e_mid = correctness_error(N, k_mid, tau)

    if e_mid <= error:
        return find_k_correct(N, tau, error, k_mid, k_end)
    else:
        return find_k_correct(N, tau, error, k_beg, k_mid)


def find_k_sec(N, tau, sec, q, k_beg=None, k_end=None):
    if k_beg is None:
        k_beg = 1
    if k_end is None:
        k_end = N - 1

    if k_beg >= k_end:
        return k_beg

    k_mid = (k_beg + k_end) // 2
    s_mid = sec_lpn(N, k_mid, tau, q)

    if s_mid < sec:
        return find_k_sec(N, tau, sec, q, k_mid + 1, k_end)
    else:
        return find_k_sec(N, tau, sec, q, k_beg, k_mid)


def find_k(N, tau, sec, error, q):
    k_max = find_k_correct(N, tau, error)
    if k_max == -1:
        return -1

    if sec_lpn(N, k_max, tau, q) >= sec:
        return find_k_sec(N, tau, sec, q, 1, k_max)
    else:
        return -1


def find_lpn_params(tau, sec, error, q, low=None, high=None):
    if low is None:
        low = 1
    if high is None:
        high = 1500

    if low >= high:
        return (low, find_k(low, tau, sec, error, q))

    N = (low + high) // 2
    k = find_k(N, tau, sec, error, q)

    if k == -1:
        return find_lpn_params(tau, sec, error, q, N + 1, high)
    else:
        return find_lpn_params(tau, sec, error, q, low, N)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate LPN params",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("sec", type=int, help="Bits of security")
    parser.add_argument("error", type=int, help="Bits of correctness error")
    parser.add_argument("tau", type=int, help="Bernoulli error bits")
    parser.add_argument("q", type=int, help="Field size")
    args = parser.parse_args()

    (N, k) = find_lpn_params(args.tau, args.sec, 2 ** (-1 * args.error), args.q)
    print(f"N={N}, k={k}, tau={args.tau}")

    if N != -1 and k != -1:
        print(f"Error: {correctness_error(N, k, args.tau)}")
        print(f"Security: {sec_lpn(N, k, args.tau, args.q)}")
