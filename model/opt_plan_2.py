"""model_2.pyをimport"""
import matplotlib.pyplot as plt
from model_2 import ProdPlan


def main(nums, indpath):
    """
    最適化問題を実行し、結果を表示する関数
    """
    # データのパスの指定
    ind_path = indpath
    # データを読み込む
    list_j, list_p, list_w, list_r = ProdPlan.pandas_read(ind_path, nums)
    # データを辞書型に変換する
    dict_p, dict_w, dict_r = ProdPlan.list_to_dict(list_j, list_p, list_w, list_r)
    # モデルのインスタンスを作成する
    prodplan = ProdPlan(list_j, dict_p, dict_w, dict_r)
    # モデルを最適化する
    prodplan.modeling()
    # モデルの求解
    prodplan.solve()
    # ガントチャートで可視化する
    prodplan.visualize()
    # 計算時間を表示する
    time = prodplan.get_time()
    # モデルの情報を表示する
    status, object_value = prodplan.get_model_info()
    # 結果を取得する
    job_order = prodplan.get_job_order()
    return status, object_value, time, job_order


if __name__ == "__main__":
    # ジョブ数を変えて計算時間の変化をみる
    time_list = []
    job_order_list = []
    num_set = range(3, 5)

    for i in num_set:
        status, object_value, time, job_order = main(i, "./model/data/")
        time_list.append(time)
        job_order_list.append(job_order)

    # resultに計算時間のグラフを保存する
    plt.plot(num_set, time_list)
    plt.xlabel("Number of jobs")
    plt.ylabel("Time [s]")
    plt.savefig("./model/result/model_2/time.pdf", format="pdf")
    # resultに計算時間を保存する
    with open("./model/result/model_2/time.csv", "w") as f:
        for i in num_set:
            f.write(str(i) + "," + str(time_list[i - 3]) + "\n")
    # resultにジョブの順序を保存する
    with open("./model/result/model_2/job_order.csv", "w") as f:
        for i in num_set:
            f.write("model" + str(i) + "," + str(job_order_list[i - 3]) + "\n")
