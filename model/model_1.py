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
        self.big_m = max(self.dict_r.values()) + sum(self.dict_p.values())

        # Model
        self.model = None
        self.var_x = dict
        self.var_c = dict
        self.var_s = dict

        # Result
        self.status = -1
        self.objective = -1

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
        self.var_c = pulp.LpVariable.dicts("c", self.jobs, lowBound=0, cat="Integer")
        self.var_s = pulp.LpVariable.dicts("s", self.jobs, lowBound=0, cat="Integer")

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

    def visualize(self, output_path):
        """
        結果をガントチャートで表示する関数
        """
        # ガントチャートの作成
        # ジョブの開始、終了の数値は日にちの値とする
        # 　ジョブの開始日は2023年1月1日とする
        gantt_chart_df = pd.DataFrame(
            {
                "Task": self.jobs,
                "Start": [
                    datetime.datetime(2023, 1, 1)
                    + datetime.timedelta(days=pulp.value(self.var_s[j]))
                    for j in self.jobs
                ],
                "Finish": [
                    datetime.datetime(2023, 1, 1)
                    + datetime.timedelta(days=pulp.value(self.var_c[j]))
                    for j in self.jobs
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
        # 保存場所はoutput_path。無い場合は作成する
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        fig.write_image(f"{output_path}/{save_name}.png")

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
        # ジョブのをスタート時間でソート
        job_order = sorted(self.jobs, key=lambda x: pulp.value(self.var_s[x]))
        return job_order
