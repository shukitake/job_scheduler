"""model_2.pyをimport"""
from model_2 import ProdPlan


def main(jobs, time_p, weights, time_r):
    """
    最適化問題を実行し、結果を表示する関数
    """
    prod_plan = ProdPlan(jobs, time_p, weights, time_r)
    prod_plan.modeling()
    prod_plan.solve()
    prod_plan.show_result()


if __name__ == "__main__":
    # 初期設定を与える
    # ジョブの集合
    Jobs = [1, 2, 3, 4]
    # ジョブの処理時間の集合
    Pro_time = [1, 93, 26, 30]
    # ジョブの重要度の集合
    Weights = [1, 3, 3, 5]
    # ジョブのリリース時間の集合
    Release_time = [63, 4, 63, 99]
    p, w, r = ProdPlan.make_data(Jobs, Pro_time, Weights, Release_time)
    main(Jobs, p, w, r)
