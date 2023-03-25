import galois
import numpy as np
import argparse
import os


def check_rs(N, T, n_bin, k_bin, d_bin):
    n_rs = (N + n_bin - 1) // n_bin
    if n_rs > 2**k_bin:
        return False

    d_rs = (n_rs * n_bin + T * d_bin - 1) // (T * d_bin)

    if d_rs > n_rs:
        return False

    k_rs = n_rs - d_rs + 1
    return (n_rs, k_rs, d_rs)


def find_bch_rs(N, T):
    best_params = None
    best_k = 0

    for m in range(1, 11):
        n_bin = 2**m - 1
        t_max = n_bin // m
        for t in np.arange(t_max, 0, -1):
            k_bin = n_bin - m * t
            if k_bin <= 1:
                continue
            d_bin = 2 * t + 1
            if d_bin < n_bin // T:
                break

            check = check_rs(N, T, n_bin + 1, k_bin, d_bin + 1)

            if check:
                n_rs, k_rs, d_rs = check
                k = k_rs * k_bin
                if k > best_k:
                    best_k = k
                    best_params = (n_rs, k_rs, d_rs, n_bin + 1, k_bin, d_bin + 1)

    return best_params


def gen_mat_rs(n_rs, k_rs, k_bch):
    assert 2**k_bch > n_rs
    GF = galois.GF(2, k_bch)

    acc = GF([1 for _ in range(n_rs)])
    base_row = GF(list(range(1, n_rs + 1)))
    mat = GF([acc])

    for _ in range(k_rs - 1):
        acc = acc * base_row
        mat = np.concatenate([mat, [acc]], axis=0)
    return mat


def gen_mat_bch(n_bch, k_bch):
    GF2 = galois.GF2
    bch = galois.BCH(n_bch - 1, k_bch, field=GF2)
    g_bch = np.concatenate((bch.G, [[GF2(1)]] * k_bch), 1)
    return g_bch


def encode_concat(g_rs, g_bch, inp):
    k_rs, _ = g_rs.shape
    k_bch, _ = g_bch.shape
    assert len(inp) == k_rs * k_bch

    GF = galois.GF(2, k_bch)
    inp_rs = GF([GF.Vector([inp[i::k_rs]]) for i in range(0, k_rs)]).transpose()

    code_rs = np.matmul(inp_rs, g_rs)
    code_concat = []
    for codeword in code_rs:
        code_concat.extend(np.matmul(codeword.vector(), g_bch))

    return galois.GF2(code_concat).flatten()


def gen_mat_concat(g_rs, g_bch):
    k_rs, _ = g_rs.shape
    k_bch, _ = g_bch.shape

    inp_len = k_rs * k_bch

    mat = []
    for i in range(inp_len):
        inp = np.zeros(inp_len, dtype=int)
        inp[i] = 1
        mat.append(encode_concat(g_rs, g_bch, inp))

    return galois.GF2(mat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate binary super invertible matrices",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("num_parties", type=int, help="Total number of parties.")
    parser.add_argument(
        "corrupt_ratio",
        type=int,
        help="Ratio of total number of parties to maximum number of corrupt parties.",
    )
    parser.add_argument(
        "--output_path",
        help="Directory to save matrix",
    )
    args = parser.parse_args()

    params = find_bch_rs(args.num_parties, args.corrupt_ratio)
    if params is None:
        print("Could not construct matrix for given params.")
    else:
        params = tuple(map(int, params))
        n_rs, k_rs, d_rs, n_bch, k_bch, d_bch = params
        print(
            f"Outer code: [{n_rs}, {k_rs}, {d_rs}] Reed-solomon code over GF(2^{k_bch})"
        )
        print(f"Inner code: [{n_bch}, {k_bch}, {d_bch}] binary BCH code")
        print(f"Concatenated code: [{n_bch * n_rs}, {k_bch * k_rs}, {d_bch * d_rs}]")
        print()

        n_cc = n_bch * n_rs
        k_cc = k_bch * k_rs
        d_cc = d_bch * d_rs
        print(f"Number of parties: {n_cc}")
        print(f"Maximum number of corruptions: {d_cc - 1}")
        print(f"Minimum number of honest parties: {n_cc - d_cc + 1}")
        print(f"Rate when used as randomness extractor: {k_cc / n_cc}")

        if args.output_path is not None:
            g_rs = gen_mat_rs(n_rs, k_rs, k_bch)
            g_bch = gen_mat_bch(n_bch, k_bch)
            g_cc = gen_mat_concat(g_rs, g_bch)
            g_cc = g_cc.T

            save_path = os.path.join(args.output_path, f"n{n_cc}_t{d_cc - 1}.txt")
            np.savetxt(save_path, g_cc, fmt="%i")
