"""最適化のモデラーをインポート"""
import datetime
import os

import pandas as pd
import plotly.express as px
import plotly.io as pio
import pulp
from utils.log import LoggerUtil


class ProdPlan:
    """ジョブスケジューリング最適化問題をモデル化するクラス

    ジョブ処理時間、ジョブリリース時間、ジョブ重みを考慮して、ジョブスケジューリング問題を最適化するための最適化モデルを作成します。

    Attributes:
    PROCESS_FNAME (str)：ジョブ処理時間が含まれるファイルの名前
    WEIGHTS_FNAME (str)：ジョブの重みが含まれるファイルの名前
    RELEASE_FNAME (str)：ジョブリリース時間が含まれるファイルの名前

    Args:
    list_j (list)：ジョブIDのリスト
    dict_p (dict)：ジョブ処理時間の辞書（キー：ジョブID、値：処理時間）
    dict_w (dict)：ジョブの重みの辞書（キー：ジョブID、値：重み）
    dict_r (dict)：ジョブリリース時間の辞書（キー：ジョブID、値：リリース時間）

    methods:
    pandas_read(indpath, nums)：pandasを使用してcsvファイルからジョブデータを読み込みます
    list_to_dict(list_j, list_p, list_w, list_r)：リストからデータを辞書に変換します
    modeling()：最適化モデルを構築し、制約を設定します
    solve()：最適化モデルを解きます
    visualize()：結果をガントチャートにして可視化します
    get_time()：処理時間を取得します
    get_model_info()：モデルの情報を取得します
    get_result()：結果の情報を取得します
    """

    PROCESS_FNAME = "process.csv"
    WEIGHTS_FNAME = "weights.csv"
    RELEASE_FNAME = "release.csv"

    @classmethod
    def pandas_read(cls, indpath, nums):
        """
        pandasでデータを読み込む関数
        """
        time_p_fpath = os.path.join(indpath, cls.PROCESS_FNAME)
        weights_fpath = os.path.join(indpath, cls.WEIGHTS_FNAME)
        release_fpath = os.path.join(indpath, cls.RELEASE_FNAME)

        time_p_dfm = pd.read_csv(time_p_fpath)
        weights_dfm = pd.read_csv(weights_fpath)
        release_dfm = pd.read_csv(release_fpath)

        list_j = list(range(1, nums + 1))
        # dfmからnums数だけ取り出してリストに変換
        list_p = time_p_dfm["p"].head(nums).to_list()
        list_w = weights_dfm["w"].head(nums).to_list()
        list_r = release_dfm["r"].head(nums).to_list()

        return list_j, list_p, list_w, list_r

    @classmethod
    def list_to_dict(cls, list_j, list_p, list_w, list_r):
        """
        データを辞書型に変換する関数
        """
        dict_p = dict()
        dict_w = dict()
        dict_r = dict()
        for v_j, v_p, v_w, v_r in zip(list_j, list_p, list_w, list_r):
            dict_p[v_j] = v_p
            dict_w[v_j] = v_w
            dict_r[v_j] = v_r
        return dict_p, dict_w, dict_r

    def __init__(self, list_j, dict_p, dict_w, dict_r):
        # ロガーの作成
        self.logger = LoggerUtil().get_logger(__name__)

        # 辞書型にして入力
        self.jobs = list_j
        self.dict_p = dict_p
        self.dict_w = dict_w
        self.dict_r = dict_r
        self.times = list(
            range(1, max(self.dict_r.values()) + sum(self.dict_p.values()))
        )

        # Model
        self.model = None
        self.var_z = dict

        # Result
        self.status = -1
        self.objective = -1

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

    def visualize(self):
        """
        結果をガントチャートで表示する関数

        z_jtの値が1のとき、ジョブjは時刻tに処理されている
        """
        # ガントチャートの作成
        # ジョブの開始、終了の数値は日にちの値とする
        # 　ジョブの開始日は2023年1月1日とする
        gantt_chart_df = pd.DataFrame(
            {
                "Task": self.jobs,
                "Start": [
                    datetime.datetime(2023, 1, 1, 0, 0, 0) + datetime.timedelta(days=t)
                    for t in [
                        pulp.value(
                            pulp.lpSum(
                                [
                                    t_dash * self.var_z[j, t_dash].value()
                                    for t_dash in self.times
                                ]
                            )
                        )
                        for j in self.jobs
                    ]
                ],
                "Finish": [
                    datetime.datetime(2023, 1, 1, 0, 0, 0) + datetime.timedelta(days=t)
                    for t in [
                        pulp.value(
                            pulp.lpSum(
                                [
                                    t_dash * self.var_z[j, t_dash].value()
                                    for t_dash in self.times
                                ]
                            )
                        )
                        + self.dict_p[j]
                        for j in self.jobs
                    ]
                ],
            }
        )

        # ガントチャートの表示
        fig = px.timeline(
            gantt_chart_df,
            x_start="Start",
            x_end="Finish",
            y="Task",
            color="Task",
            title="Job Shop Scheduling",
        )
        fig.update_yaxes(autorange="reversed")
        save_name = "gantt_chart_sample"
        pio.orca.config.executable = "/Applications/orca.app/Contents/MacOS/orca"
        # 画像の保存
        # 保存場所はresult/model_1
        pio.write_image(fig, f"./model/result/model_3/{save_name}.png")

    def get_time(self):
        """
        処理時間を取得する関数
        """
        return self.model.solutionTime

    def get_model_info(self):
        """
        モデルの情報を取得する関数
        """
        return pulp.LpStatus[self.status], self.model.objective.value()

    def get_job_order(self):
        """
        ジョブの順番を取得する関数
        """
        # ジョブをスタート時間でソート
        job_order = sorted(
            [
                [
                    j,
                    pulp.value(
                        pulp.lpSum(
                            [
                                t_dash * self.var_z[j, t_dash].value()
                                for t_dash in self.times
                            ]
                        )
                    ),
                ]
                for j in self.jobs
            ],
            key=lambda x: x[1],
        )

        return job_order
