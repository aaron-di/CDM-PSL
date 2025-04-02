import torch
import numpy as np


def get(name, *args, **kwargs):
    name = name.lower()

    PROBLEM = {
        "re1": RE1,
        "re2": RE2,
        "re3": RE3,
        "re4": RE4,
        "re5": RE5,
        "re6": RE6,
        "re7": RE7,
    }

    if name not in PROBLEM:
        raise Exception("Problem not found.")

    return PROBLEM[name](*args, **kwargs)


class RE1:
    def __init__(self, n_dim=4):
        F = 10.0
        sigma = 10.0
        tmp_val = F / sigma

        self.n_var = n_dim
        self.n_obj = 2
        self.lbound = torch.tensor(
            [tmp_val, np.sqrt(2.0) * tmp_val, np.sqrt(2.0) * tmp_val, tmp_val]
        ).float()
        self.ubound = torch.ones(n_dim).float() * 3 * tmp_val
        self.nadir_point = [2886.3695604236013, 0.039999999999998245]

    def evaluate(self, x):
        x = torch.from_numpy(x).to("cuda")
        F = 10.0
        E = 2.0 * 1e5
        L = 200.0

        if x.device.type == "cuda":
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        x = x * (self.ubound - self.lbound) + self.lbound

        f1 = L * (
            (2 * x[:, 0]) + np.sqrt(2.0) * x[:, 1] + torch.sqrt(x[:, 2]) + x[:, 3]
        )
        f2 = ((F * L) / E) * (
            (2.0 / x[:, 0])
            + (2.0 * np.sqrt(2.0) / x[:, 1])
            - (2.0 * np.sqrt(2.0) / x[:, 2])
            + (2.0 / x[:, 3])
        )

        f1 = f1
        f2 = f2

        objs = torch.stack([f1, f2]).T
        return objs.cpu().numpy()


class RE2:
    def __init__(self, n_dim=4):
        self.n_var = n_dim
        self.n_obj = 2
        self.lbound = torch.tensor([1, 1, 10, 10]).float()
        self.ubound = torch.tensor([100, 100, 200, 240]).float()
        self.nadir_point = [5852.05896876, 1288669.78054]

    def evaluate(self, x):
        x = torch.from_numpy(x).to("cuda")
        if x.device.type == "cuda":
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        x = x * (self.ubound - self.lbound) + self.lbound

        x1 = 0.0625 * torch.round(x[:, 0])
        x2 = 0.0625 * torch.round(x[:, 1])
        x3 = x[:, 2]
        x4 = x[:, 3]

        # First original objective function
        f1 = (
            (0.6224 * x1 * x3 * x4)
            + (1.7781 * x2 * x3 * x3)
            + (3.1661 * x1 * x1 * x4)
            + (19.84 * x1 * x1 * x3)
        )
        f1 = f1.float()

        # Original constraint functions
        g1 = x1 - (0.0193 * x3)
        g2 = x2 - (0.00954 * x3)
        g3 = (np.pi * x3 * x3 * x4) + ((4.0 / 3.0) * (np.pi * x3 * x3 * x3)) - 1296000

        g = torch.stack([g1, g2, g3])
        z = torch.zeros(g.shape).cuda().to(torch.float64)
        g = torch.where(g < 0, -g, z)

        f2 = torch.sum(g, axis=0).to(torch.float64)

        objs = torch.stack([f1, f2]).T

        return objs.cpu().numpy()


class RE3:
    def __init__(self, n_dim=4):
        self.n_var = n_dim
        self.n_obj = 3
        self.lbound = torch.tensor([55, 75, 1000, 11]).float()
        self.ubound = torch.tensor([80, 110, 3000, 20]).float()
        self.nadir_point = [5.3067, 3.12833430979, 25.0]

    def evaluate(self, x):
        x = torch.from_numpy(x).to("cuda")
        if x.device.type == "cuda":
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        x = x * (self.ubound - self.lbound) + self.lbound

        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]

        # First original objective function
        f1 = 4.9 * 1e-5 * (x2 * x2 - x1 * x1) * (x4 - 1.0)
        # Second original objective function
        f2 = ((9.82 * 1e6) * (x2 * x2 - x1 * x1)) / (
            x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1)
        )

        # Reformulated objective functions
        g1 = (x2 - x1) - 20.0
        g2 = 0.4 - (x3 / (3.14 * (x2 * x2 - x1 * x1)))
        g3 = 1.0 - (2.22 * 1e-3 * x3 * (x2 * x2 * x2 - x1 * x1 * x1)) / torch.pow(
            (x2 * x2 - x1 * x1), 2
        )
        g4 = (2.66 * 1e-2 * x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1)) / (
            x2 * x2 - x1 * x1
        ) - 900.0

        g = torch.stack([g1, g2, g3, g4])
        z = torch.zeros(g.shape).float().cuda().to(torch.float64)
        g = torch.where(g < 0, -g, z)

        f3 = torch.sum(g, axis=0).float()

        objs = torch.stack([f1, f2, f3]).T

        return objs.cpu().numpy()


class RE4:
    def __init__(self, n_dim=4):
        self.n_var = n_dim
        self.n_obj = 3
        self.lbound = torch.tensor([12, 12, 12, 12]).float()
        self.ubound = torch.tensor([60, 60, 60, 60]).float()
        self.nadir_point = [5.931, 56.0, 0.355720675227]

    def evaluate(self, x):
        x = torch.from_numpy(x).to("cuda")
        if x.device.type == "cuda":
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        x = x * (self.ubound - self.lbound) + self.lbound

        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]

        # First original objective function
        f1 = torch.abs(6.931 - ((x3 / x1) * (x4 / x2)))
        # Second original objective function (the maximum value among the four variables)
        l = torch.stack([x1, x2, x3, x4])
        f2 = torch.max(l, dim=0)[0]

        g1 = 0.5 - (f1 / 6.931)

        g = torch.stack([g1])
        z = torch.zeros(g.shape).float().cuda().to(torch.float64)
        g = torch.where(g < 0, -g, z)
        f3 = g[0]

        objs = torch.stack([f1, f2, f3]).T

        return objs.cpu().numpy()


class RE5:
    def __init__(self, n_dim=4):
        self.n_var = n_dim
        self.n_obj = 3
        self.lbound = torch.tensor([0, 0, 0, 0]).float()
        self.ubound = torch.tensor([1, 1, 1, 1]).float()
        self.nadir_point = [0.98949120096, 0.956587924661, 0.987530948586]

    def evaluate(self, x):
        x = torch.from_numpy(x).to("cuda")
        if x.device.type == "cuda":
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        x = x * (self.ubound - self.lbound) + self.lbound

        xAlpha = x[:, 0]
        xHA = x[:, 1]
        xOA = x[:, 2]
        xOPTT = x[:, 3]

        # f1 (TF_max)
        f1 = (
            0.692
            + (0.477 * xAlpha)
            - (0.687 * xHA)
            - (0.080 * xOA)
            - (0.0650 * xOPTT)
            - (0.167 * xAlpha * xAlpha)
            - (0.0129 * xHA * xAlpha)
            + (0.0796 * xHA * xHA)
            - (0.0634 * xOA * xAlpha)
            - (0.0257 * xOA * xHA)
            + (0.0877 * xOA * xOA)
            - (0.0521 * xOPTT * xAlpha)
            + (0.00156 * xOPTT * xHA)
            + (0.00198 * xOPTT * xOA)
            + (0.0184 * xOPTT * xOPTT)
        )
        # f2 (X_cc)
        f2 = (
            0.153
            - (0.322 * xAlpha)
            + (0.396 * xHA)
            + (0.424 * xOA)
            + (0.0226 * xOPTT)
            + (0.175 * xAlpha * xAlpha)
            + (0.0185 * xHA * xAlpha)
            - (0.0701 * xHA * xHA)
            - (0.251 * xOA * xAlpha)
            + (0.179 * xOA * xHA)
            + (0.0150 * xOA * xOA)
            + (0.0134 * xOPTT * xAlpha)
            + (0.0296 * xOPTT * xHA)
            + (0.0752 * xOPTT * xOA)
            + (0.0192 * xOPTT * xOPTT)
        )
        # f3 (TT_max)
        f3 = (
            0.370
            - (0.205 * xAlpha)
            + (0.0307 * xHA)
            + (0.108 * xOA)
            + (1.019 * xOPTT)
            - (0.135 * xAlpha * xAlpha)
            + (0.0141 * xHA * xAlpha)
            + (0.0998 * xHA * xHA)
            + (0.208 * xOA * xAlpha)
            - (0.0301 * xOA * xHA)
            - (0.226 * xOA * xOA)
            + (0.353 * xOPTT * xAlpha)
            - (0.0497 * xOPTT * xOA)
            - (0.423 * xOPTT * xOPTT)
            + (0.202 * xHA * xAlpha * xAlpha)
            - (0.281 * xOA * xAlpha * xAlpha)
            - (0.342 * xHA * xHA * xAlpha)
            - (0.245 * xHA * xHA * xOA)
            + (0.281 * xOA * xOA * xHA)
            - (0.184 * xOPTT * xOPTT * xAlpha)
            - (0.281 * xHA * xAlpha * xOA)
        )

        objs = torch.stack([f1, f2, f3]).T

        return objs.cpu().numpy()


def div(x1, x2):

    return np.divide(x1, x2, out=np.zeros(np.broadcast(x1, x2).shape), where=(x2 != 0))


class RE6:
    """
    Reinforced concrete beam design
    """

    def __init__(self, n_dim=3):

        self.n_var = 3
        self.n_obj = 2
        self.lbound = torch.tensor([0.2, 0, 0]).float()
        self.ubound = torch.tensor([15, 20, 40]).float()
        self.nadir_point = [0.98949120096, 0.956587924661, 0.987530948586]

    feasible_values = np.array(
        [
            0.20,
            0.31,
            0.40,
            0.44,
            0.60,
            0.62,
            0.79,
            0.80,
            0.88,
            0.93,
            1.0,
            1.20,
            1.24,
            1.32,
            1.40,
            1.55,
            1.58,
            1.60,
            1.76,
            1.80,
            1.86,
            2.0,
            2.17,
            2.20,
            2.37,
            2.40,
            2.48,
            2.60,
            2.64,
            2.79,
            2.80,
            3.0,
            3.08,
            3,
            10,
            3.16,
            3.41,
            3.52,
            3.60,
            3.72,
            3.95,
            3.96,
            4.0,
            4.03,
            4.20,
            4.34,
            4.40,
            4.65,
            4.74,
            4.80,
            4.84,
            5.0,
            5.28,
            5.40,
            5.53,
            5.72,
            6.0,
            6.16,
            6.32,
            6.60,
            7.11,
            7.20,
            7.80,
            7.90,
            8.0,
            8.40,
            8.69,
            9.0,
            9.48,
            10.27,
            11.0,
            11.06,
            11.85,
            12.0,
            13.0,
            14.0,
            15.0,
        ]
    )

    def evaluate(self, x):
        x = torch.from_numpy(x).to("cuda")
        if x.device.type == "cuda":
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        x = x * (self.ubound - self.lbound) + self.lbound
        x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
        x1 = self.closest_value(self.feasible_values, x1.cpu().numpy())
        x1 = torch.tensor(x1).to("cuda").float()
        f1 = (29.4 * x1) + (0.6 * x2 * x3)
        x1 = x1.cpu().numpy()
        x2 = x2.cpu().numpy()
        x3 = x3.cpu().numpy()
        g = np.column_stack(
            [(x1 * x3) - 7.735 * div((x1 * x1), x2) - 180.0, 4.0 - div(x3, x2)]
        )

        g[g >= 0] = 0
        g[g < 0] = -g[g < 0]

        f2 = np.sum(g, axis=1)
        f2 = torch.tensor(f2).to("cuda").float()
        objs = torch.stack([f1, f2]).T
        return objs.cpu().numpy()

    def closest_value(self, arr, val):

        return arr[np.argmin(np.abs(arr[:, None] - val), axis=0)]


class RE7:
    def __init__(self, n_dim=4):

        self.n_var = 4
        self.n_obj = 3
        self.lbound = torch.tensor([0.125, 0.1, 0.1, 0.125]).float()
        self.ubound = torch.tensor([5, 10, 10, 5]).float()
        self.nadir_point = [0.98949120096, 0.956587924661, 0.987530948586]

    def evaluate(self, x):
        x = torch.from_numpy(x).to("cuda")
        if x.device.type == "cuda":
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        x = x * (self.ubound - self.lbound) + self.lbound
        x1, x2, x3, x4 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]

        P = 6000
        L = 14
        E = 30 * 1e6
        G = 12 * 1e6
        tauMax = 13600
        sigmaMax = 30000
        x1 = x1.cpu().numpy()
        x2 = x2.cpu().numpy()
        x3 = x3.cpu().numpy()
        x4 = x4.cpu().numpy()
        f1 = (1.10471 * x1 * x1 * x2) + (0.04811 * x3 * x4) * (14.0 + x2)
        f2 = div(4 * P * L * L * L, E * x4 * x3 * x3 * x3)

        M = P * (L + (x2 / 2))
        tmpVar = ((x2 * x2) / 4.0) + np.power((x1 + x3) / 2.0, 2)
        R = np.sqrt(tmpVar)
        tmpVar = ((x2 * x2) / 12.0) + np.power((x1 + x3) / 2.0, 2)
        J = 2 * np.sqrt(2) * x1 * x2 * tmpVar

        tauDashDash = div(M * R, J)
        tauDash = div(P, np.sqrt(2) * x1 * x2)
        tmpVar = (
            tauDash * tauDash
            + div((2 * tauDash * tauDashDash * x2), (2 * R))
            + (tauDashDash * tauDashDash)
        )
        tau = np.sqrt(tmpVar)
        sigma = div(6 * P * L, x4 * x3 * x3)
        tmpVar = (
            4.013
            * E
            * np.sqrt((x3 * x3 * x4 * x4 * x4 * x4 * x4 * x4) / 36.0)
            / (L * L)
        )
        tmpVar2 = (x3 / (2 * L)) * np.sqrt(E / (4 * G))
        PC = tmpVar * (1 - tmpVar2)

        g = np.column_stack([tauMax - tau, sigmaMax - sigma, x4 - x1, PC - P])

        g[g >= 0] = 0
        g[g < 0] = -g[g < 0]

        f3 = np.sum(g, axis=1)
        f1 = torch.tensor(f1).to("cuda").float()
        f2 = torch.tensor(f2).to("cuda").float()
        f3 = torch.tensor(f3).to("cuda").float()
        objs = torch.stack([f1, f2, f3]).T
        return objs.cpu().numpy()
