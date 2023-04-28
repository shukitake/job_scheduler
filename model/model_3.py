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
    T : int
        時間の集合。最大値はリリース時間の最大値 + 処理時間の合計
    """

    def __init__(self, jobs, time_p, weights, time_r) -> None:
        # ロガーの作成
        self.logger = LoggerUtil().get_logger(__name__)

        # 辞書型にして入力
        self.jobs = jobs
        self.dict_p = time_p
        self.dict_w = weights
        self.dict_r = time_r
        self.times = list(
            range(1, max(self.dict_r.values()) + sum(self.dict_p.values()))
        )

        # Model
        self.model = None
        self.var_z = dict

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

        制約1 : sum_z_j = 1
        任意のジョブは一度だけ処理される
        制約2 : sum_j(p_j * z_j) <= T
        ジョブは同時に一つのみ処理される
        """
        # 　モデルのインスタンスの作成
        self.model = pulp.LpProblem(name="model", sense=pulp.LpMinimize)

        # 変数の作成
        self.var_z = pulp.LpVariable.dicts(
            "z",
            [(j, t) for j in self.jobs for t in self.times],
            cat="Binary",
        )

        # 制約の作成
        # 制約1
        for j in self.jobs:
            self.model += (
                pulp.lpSum(
                    [
                        self.var_z[j, t]
                        for t in list(
                            range(self.dict_r[j], max(self.times) - self.dict_p[j] + 1)
                        )
                    ]
                )
                == 1
            )

        # 制約2
        for t in self.times:
            self.model += (
                pulp.lpSum(
                    [
                        [
                            self.var_z[j, t_dash]
                            for t_dash in list(
                                range(max(1, t - self.dict_p[j] + 1), t + 1)
                            )
                        ]
                        for j in self.jobs
                    ]
                )
                <= 1
            )

        # 目的関数の作成
        self.model += pulp.lpSum(
            [
                self.dict_w[j]
                * (
                    pulp.lpSum([t * self.var_z[j, t] for t in self.times])
                    * self.dict_p[j]
                )
                for j in self.jobs
            ]
        )
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
        # 結果の表示
        for j in self.jobs:
            for t in self.times:
                if self.var_z[j, t].value() == 1:
                    self.logger.info(f"ジョブ{j}は時刻{t}に処理される")
        return
