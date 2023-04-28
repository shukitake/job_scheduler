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
    """

    def __init__(self, jobs, time_p, weights, time_r) -> None:
        # ロガーの作成
        self.logger = LoggerUtil().get_logger(__name__)

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

        切除平面法を用いて最適化問題を解く
        最初に追加する制約は以下の通り
        1. 全てのジョブは終了時間がリリース時間と処理時間の合計よりも大きい
        """
        # 　モデルのインスタンスの作成
        self.model = pulp.LpProblem(name="model", sense=pulp.LpMinimize)

        # 変数の作成
        self.var_x = pulp.LpVariable.dicts(
            "x", [(j, k) for j in self.jobs for k in self.jobs], cat="Binary"
        )
        self.var_c = pulp.LpVariable.dicts("C", self.jobs, lowBound=0, cat="Integer")
        self.var_s = pulp.LpVariable.dicts("S", self.jobs, lowBound=0, cat="Integer")

        # 制約の作成
        for j in self.jobs:
            self.model += self.var_c[j] >= self.dict_r[j] + self.dict_p[j]

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