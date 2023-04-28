"""最適化のモデラーをインポート"""
import pulp
from utils.log import LoggerUtil


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
    M : int
        ビックMの値（最も遅いリリースから全ジョブの処理時間の和を足す）
    """

    def __init__(self, jobs, time_p, weights, time_r) -> None:
        # ロガーの作成
        self.logger = LoggerUtil().get_logger(__name__)

        # 辞書型にして入力
        self.jobs = jobs
        self.dict_p = time_p
        self.dict_w = weights
        self.dict_r = time_r
        self.big_m = max(self.dict_r.values()) + sum(self.dict_p.values())

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

        制約1 : C[j] == S[j] + P[j]
            ジョブjの完了時間は開始時間と処理時間の和
        制約2 : S[j] >= R[j]
            ジョブjの開始時間はリリース時間以上
        制約3 : C[j] <= S[k] + M * (1 - X[j,k])
            ジョブjがジョブkよりも早く終わる時、ジョブjの完了時間はジョブkの開始時間よりも早い
        制約4 : X[j,k] + X[k,j] == 1
            ジョブjとジョブkの順序は1つのみ
        """
        # 　モデルのインスタンスの作成
        self.model = pulp.LpProblem(name="model", sense=pulp.LpMinimize)

        # 変数の作成
        self.var_x = pulp.LpVariable.dicts(
            "x", [(j, k) for j in self.jobs for k in self.jobs], cat="Binary"
        )
        self.var_c = pulp.LpVariable.dicts("C", self.jobs, lowBound=0, cat="Integer")
        self.var_s = pulp.LpVariable.dicts("S", self.jobs, lowBound=0, cat="Integer")

        # 制約条件を定義する
        for j in self.jobs:
            # 制約1
            self.model += self.var_c[j] == self.var_s[j] + self.dict_p[j]
            # 制約2
            self.model += self.var_s[j] >= self.dict_r[j]
            for k in self.jobs:
                # 制約3
                self.model += (
                    self.var_s[k] + self.big_m * (1 - self.var_x[j, k]) >= self.var_c[j]
                )
                # 制約4
                if j != k:
                    self.model += self.var_x[j, k] + self.var_x[k, j] == 1
        # 目的関数の作成
        self.model += pulp.lpSum([self.dict_w[j] * self.var_c[j] for j in self.jobs])
        return

    def solve(self):
        """
        最適化計算を行う関数
        """
        # 最適化計算とステータスの取得
        self.status = self.model.solve()
        self.logger.info(f"Status{pulp.LpStatus[self.status]}:")

        # 目的関数値の取得
        self.objective = self.model.objective.value()

        # modelの制約の数
        self.logger.info(f"制約の数 : {self.model.numConstraints()}")
        # modelの変数の数
        self.logger.info(f"変数の数 : {self.model.numVariables()}")
        # 最適解
        self.logger.info(f"最適値 : {pulp.value(self.model.objective)}")

    def show_result(self):
        """
        結果を表示する関数
        """
        for j in self.jobs:
            self.logger.info(f"ジョブ,{j},の開始時間は,{pulp.value(self.var_s[j])}")
            self.logger.info(f"ジョブ{j}の完了時間は,{pulp.value(self.var_c[j])}")
            self.logger.info("")
        return
