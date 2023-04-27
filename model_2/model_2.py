"""最適化のモデラーをインポート"""
import pulp


class ProdPlan:
    """
    モデルを作成するクラス

    Attributes
    ----------
    J : list
        ジョブの集合
    P : dict
        ジョブの処理時間の集合
    W : dict
        ジョブの重要度の集合
    R : dict
        ジョブのリリース時間の集合
    """

    def __init__(self, jobs, time_p, weights, time_r) -> None:
        # 辞書型にして入力
        self.jobs = jobs
        self.dict_p = time_p
        self.dict_w = weights
        self.dict_r = time_r

        # Model
        self.model = None
        self.var_x = dict
        self.var_c = dict
        self.var_s = dict

        # Result
        self.status = -1
        self.objective = -1

    @classmethod
    def make_data(cls, jobs, time_p, weights, time_r):
        """
        データを辞書型に変換する関数
        """
        dict_p = dict()
        dict_w = dict()
        dict_r = dict()
        for j, p_value, w_value, r_value in zip(jobs, time_p, weights, time_r):
            dict_p[j] = p_value
            dict_w[j] = w_value
            dict_r[j] = r_value
        return dict_p, dict_w, dict_r

    def modeling(self):
        """
        モデルを作成する関数

        制約1 : C[j] == s[j] + p[j]
        ジョブjの完了時間は開始時間と処理時間の和
        制約2 : s[j] >= r[j]
        ジョブjの開始時間はリリース時間以上
        制約3 : s[j] >= r[j] * X[k,j] sum_j p[i] *(x[i,j] - x[i,k])

        制約4 : x[j,k] + x[k,j] = 1
        ジョブjとジョブkは同時に処理されない
        制約5 : x[j,k] + x[k,i] + x[i,j]= 1
        線型順序制約
        """
        # 　モデルのインスタンスの作成
        self.model = pulp.LpProblem(name="model", sense=pulp.LpMinimize)

        # 変数の作成
        self.var_x = pulp.LpVariable.dicts(
            "x", [(j, k) for j in self.jobs for k in self.jobs], cat="Binary"
        )
        self.var_c = pulp.LpVariable.dicts("C", self.jobs, lowBound=0, cat="Continuous")
        self.var_s = pulp.LpVariable.dicts("S", self.jobs, lowBound=0, cat="Continuous")

        # 制約の作成
        for j in self.jobs:
            self.model += self.var_c[j] == self.var_s[j] + self.dict_p[j]
            self.model += self.var_s[j] >= self.dict_r[j]
            for k in self.jobs:
                self.model += self.var_s[j] >= self.dict_r[j] * self.var_x[
                    (k, j)
                ] + pulp.lpSum(
                    [
                        self.dict_p[j] * (self.var_x[(i, j)] - self.var_x[(i, k)])
                        for i in self.jobs
                    ]
                )
                if j != k:
                    self.model += self.var_x[(j, k)] + self.var_x[(k, j)] == 1
                for i in self.jobs:
                    self.model += (
                        self.var_x[(j, k)] + self.var_x[(k, i)] + self.var_x[(i, j)]
                        <= 1
                    )

        # 目的関数の作成
        self.model += pulp.lpSum([self.dict_w[j] * self.var_c[j] for j in self.jobs])
        return

    def solve(self):
        """
        最適化計算を行う関数
        """
        # 最適化計算とステータスの取得
        self.status = self.model.solve()
        print(f"Status{pulp.LpStatus[self.status]}:")

        # 目的関数値の取得
        self.objective = self.model.objective.value()

        # modelの制約の数
        print("制約の数 : ", self.model.numConstraints())
        # modelの変数の数
        print("変数の数 : ", self.model.numVariables())
        # 最適解
        print("最適解 : ", pulp.value(self.model.objective))
        # 最適解の時の変数の値
        for v in self.model.variables():
            print(v.name, "=", v.varValue)
        return
