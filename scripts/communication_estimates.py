import math
import argparse


class Params:
    def __init__(
        self,
        field_bits,
        num_parties,
        threshold,
        pack,
        binsup_dim,
        lpn_key_len,
        lpn_ctx_len,
        lpn_noise_param,
        lenmac,
        lencheck,
    ):
        self.f = field_bits
        self.fcomm = (field_bits + 7) // 8
        self.n = num_parties
        self.t = threshold
        self.l = pack
        self.binsup_dim = binsup_dim
        self.lpn_key_len = lpn_key_len
        self.lpn_ctx_len = lpn_ctx_len
        self.lpn_noise_param = lpn_noise_param
        self.lenmac = lenmac
        self.lencheck = lencheck


class ProtoCtr:
    def __init__(self, params):
        self.counts = {}
        self.params = params
        self.ncons = 0
        self.nmac = 0
        self.nzero = 0

    def __inc(self, protocol, num):
        self.counts[protocol] = self.counts.get(protocol, 0) + num

    def allrecon(self, num):
        # Each party sends its num shares to every other party (no broadcast channel).
        return self.params.n * num * self.params.n * self.params.fcomm

    def allshare(self, num):
        # Each party deals num shares to every other party.
        return self.params.n * num * self.params.n * self.params.fcomm

    def rand(self, num):
        self.__inc("rand", num)
        return 0

    def __rand(self, num):
        batch_size = self.params.n - self.params.t
        batches = int(math.ceil(num / batch_size))
        return self.allshare(batches)

    def zero(self, num):
        self.__inc("zero", num)
        return 0

    def __zero(self, num):
        batch_size = self.params.n - self.params.t
        batches = int(math.ceil(num / batch_size))
        return self.allshare(batches)

    def bitrand(self, num):
        self.__inc("bitrand", num)
        return 0

    def __bitrand(self, num):
        batch_size = self.params.binsup_dim
        batches = int(math.ceil(num / batch_size))
        return self.allshare(batches)

    def degreduce(self, num):
        return num * 2 * self.params.n * self.params.fcomm

    def randshare(self, num):
        n = self.params.n
        l = self.params.l
        t = self.params.t

        batch_size = l
        batches = int(math.ceil(num / batch_size))

        comm = self.zero(batches * 2 * n)
        comm += self.rand(batches * (n + t))
        comm += self.allrecon(2 * batches)
        return comm

    def mackeygen(self, num):
        self.__inc("mackeygen", num)
        return 0

    def __mackeygen(self, num):
        batch_size = 2 * self.params.l
        batches = int(math.ceil(num / batch_size))

        return (
            self.allshare(batches)
            + batches * self.params.n * 2 * self.params.t * self.params.fcomm
        )

    def coin(self):
        # Reconstruct AES-128 key.
        num = int(math.ceil(128 / self.params.f))
        return self.rand(num) + self.allrecon(num)

    def mult(self, num):
        return self.rand(num) + self.zero(num) + self.degreduce(num)

    def authmult(self, num):
        return self.mult((self.params.lenmac + 1) * num)

    def trans(self, num):
        return self.randshare(num) + self.degreduce(num)

    def authtrans(self, num):
        return self.trans((self.params.lenmac + 1) * num)

    def auth(self, num):
        return self.mult(self.params.lenmac * num)

    def error(self, num):
        return (
            self.bitrand(num * self.params.lpn_noise_param)
            + self.mult(num * (self.params.lpn_noise_param - 1))
            + self.rand(num)
            + self.mult(num)
        )

    def checkbit(self, num):
        return self.authmult(num)

    def autherror(self, num):
        return (
            self.bitrand(num * self.params.lpn_noise_param)
            + self.auth(num * self.params.lpn_noise_param)
            + self.checkbit(num * self.params.lpn_noise_param)
            + self.authmult(num * (self.params.lpn_noise_param - 1))
            + self.rand(num)
            + self.mult(num * (self.params.lenmac + 1))
        )

    def checkcons(self):
        return (
            self.rand(self.params.lencheck)
            + self.coin()
            + self.allrecon(self.params.lencheck)
        )

    def checkmac(self):
        return (
            self.coin()
            + self.allrecon(self.params.lenmac)
            + self.allrecon(self.params.lenmac)
        )

    def checkzero(self):
        return self.coin() + self.allrecon(self.params.lencheck)

    def get_circ_ind_preproc(self):
        lookup = {
            "rand": self.__rand,
            "zero": self.__zero,
            "bitrand": self.__bitrand,
            "mackeygen": self.__mackeygen,
        }

        res = {}

        for k, v in self.counts.items():
            if k not in lookup:
                raise Exception(f"Unknown protocol encountered: {k}")

            res[k] = lookup[k](v)

        return res


def fmt_size(num, suffix="B"):
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(num) < 1000.0:
            return f"{num:.3f} {unit}{suffix}"
        num /= 1000.0
    return f"{num:.1f} Y{suffix}"


def get_print_stat(n):
    def print_stat(label, val):
        print(f"{label}: {fmt_size(val / n)}")

    return print_stat


def malicious_protocol(params, num_inp, num_and, num_xor, print_stat):
    get_blocks = lambda x: int(math.ceil(x / params.l))
    num_and_blocks = get_blocks(num_and)
    num_xor_blocks = get_blocks(num_xor)
    num_wire_blocks = get_blocks(num_inp) + num_and_blocks + num_xor_blocks
    num_gate_blocks = num_and_blocks + num_xor_blocks

    ctr = ProtoCtr(params)
    total = 0
    preproc = {}

    print("--- Garbling phase ---")

    # Preprocessing
    comm = ctr.mackeygen(params.lenmac)
    preproc["Generate MAC keys"] = comm

    comm = ctr.rand(num_wire_blocks * 2 * params.lpn_key_len)
    comm += ctr.auth(num_wire_blocks * 2 * params.lpn_key_len)
    preproc["Generate keys"] = comm

    comm = ctr.bitrand(num_wire_blocks)
    comm += ctr.auth(num_wire_blocks)
    comm += ctr.checkbit(num_wire_blocks)
    preproc["Generate masks"] = comm

    comm = ctr.autherror(num_gate_blocks * 4 * params.lpn_ctx_len)
    preproc["Generate errors"] = comm

    # Garbling
    comm = ctr.authtrans(num_wire_blocks * (2 * params.lpn_key_len + 1))
    comm += ctr.authtrans(2 * num_gate_blocks * (2 * params.lpn_key_len + 1))
    ctr.nmac += 2 * num_gate_blocks + 2 * num_gate_blocks * 2 * params.lpn_key_len
    print_stat("Transform masks and labels", comm)
    total += comm

    # Select plaintext to encrypt for each row of the garbled table.
    num_plaintext_selection_mult = num_and_blocks * (
        3 * (params.lpn_key_len + 1) + 1
    ) + num_xor_blocks * (params.lpn_key_len + 1)
    comm = ctr.authmult(num_plaintext_selection_mult)
    print_stat("Compute plaintexts", comm)
    total += comm

    # The actual protocol requires a degree reduction, but an auth mult is identical.
    comm = ctr.authmult(4 * num_gate_blocks * params.lpn_ctx_len)
    print_stat("Compute ciphertexts", comm)
    total += comm

    comm = ctr.checkcons()
    comm += ctr.checkmac()
    comm += ctr.checkzero()
    print_stat("Verification", comm)
    total += comm

    print_stat("Total", total)

    print("\n--- Preprocessing phase ---")
    total = 0

    for k, v in preproc.items():
        print_stat(f"{k}", v)
        total += v

    for k, v in ctr.get_circ_ind_preproc().items():
        print_stat(f"{k}", v)
        total += v

    print_stat("Total", total)


def semihon_protocol(params, num_inp, num_and, num_xor, print_stat):
    get_blocks = lambda x: int(math.ceil(x / params.l))
    num_and_blocks = get_blocks(num_and)
    num_xor_blocks = get_blocks(num_xor)
    num_wire_blocks = get_blocks(num_inp) + num_and_blocks + num_xor_blocks
    num_gate_blocks = num_and_blocks + num_xor_blocks

    ctr = ProtoCtr(params)
    total = 0
    preproc = {}

    print("--- Garbling Phase ---")

    # Preprocessing
    comm = ctr.rand(num_wire_blocks * 2 * params.lpn_key_len)
    preproc["Generate keys"] = comm

    comm = ctr.bitrand(num_wire_blocks)
    preproc["Generate masks"] = comm

    comm = ctr.error(num_gate_blocks * 4 * params.lpn_ctx_len)
    preproc["Generate errors"] = comm

    # Garbling
    comm = ctr.trans(num_wire_blocks + num_wire_blocks * 2 * params.lpn_key_len)
    comm += ctr.trans(
        2 * num_gate_blocks + 2 * num_gate_blocks * 2 * params.lpn_key_len
    )
    print_stat("Transform masks and labels", comm)
    total += comm

    num_plaintext_selection_mult = num_and_blocks * (
        3 * (params.lpn_key_len + 1) + 1
    ) + num_xor_blocks * (params.lpn_key_len + 1)
    comm = ctr.mult(num_plaintext_selection_mult)
    print_stat("Compute plaintexts", comm)
    total += comm

    print_stat("Total", total)

    print("\n--- Preprocessing Phase ---")
    total = 0

    for k, v in preproc.items():
        print_stat(f"{k}", v)
        total += v

    for k, v in ctr.get_circ_ind_preproc().items():
        print_stat(f"{k}", v)
        total += v

    print_stat("Total", total)


def cli_args():
    parser = argparse.ArgumentParser(
        description="Computation costs for each phase of the protocol.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "num_inp",
        type=int,
        help="Total number of INPUT wires for the circuit.",
    )
    parser.add_argument(
        "num_and",
        type=int,
        help="Total number of AND gates in the circuit.",
    )
    parser.add_argument(
        "num_xor",
        type=int,
        help="Total number of XOR gates in the circuit.",
    )

    parser.add_argument(
        "-n", "--num", type=int, help="Number of parties", required=True
    )
    parser.add_argument(
        "-t", "--threshold", type=int, help="Corruption threshold", required=True
    )
    parser.add_argument(
        "-l", "--packing-param", type=int, help="Packing parameter", required=True
    )
    parser.add_argument(
        "--field-bits", type=int, help="Size of each field element in bits", default=18
    )
    parser.add_argument(
        "--binsup-dim", type=int, help="Packing parameter", required=False
    )
    parser.add_argument("--lpn-key-len", type=int, default=127, help="LPN key length")
    parser.add_argument(
        "--lpn-ctx-len",
        type=int,
        default=555,
        help="LPN ciphertext length",
    )
    parser.add_argument(
        "--lpn-error-bias",
        type=int,
        default=2,
        help="Base 2 log of LPN bernoulli error probability",
    )
    parser.add_argument(
        "--lenmac", type=int, default=3, help="Number of MACs per share"
    )
    parser.add_argument(
        "--lencheck",
        type=int,
        default=3,
        help="Number of repitions for degree and zero check",
    )

    parser.add_argument(
        "--time-add",
        type=float,
        default=5.6319e-10,
        help="Time for a field addition in seconds",
    )
    parser.add_argument(
        "--time-mult",
        type=float,
        default=1.0079e-8,
        help="Time for a field multiplication in seconds",
    )
    parser.add_argument(
        "--semi-honest",
        action=argparse.BooleanOptionalAction,
        help="Estimate the cost of semi-honest protoocol (default is malicious)",
    )
    parser.add_argument(
        "--threads", type=int, default=1, help="Number of threads per party"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = cli_args()

    binsup_dim = args.binsup_dim
    if binsup_dim is None:
        binsup_dim = {
            (128, 31): 14,
            (256, 63): 42,
            (512, 127): 77,
        }[(args.num, args.threshold)]

    params = Params(
        args.field_bits,
        args.num,
        args.threshold,
        args.packing_param,
        binsup_dim,
        args.lpn_key_len,
        args.lpn_ctx_len,
        args.lpn_error_bias,
        args.lenmac,
        args.lencheck,
    )

    print_stat = get_print_stat(args.num)
    if args.semi_honest:
        semihon_protocol(params, args.num_inp, args.num_and, args.num_xor, print_stat)
    else:
        malicious_protocol(params, args.num_inp, args.num_and, args.num_xor, print_stat)
