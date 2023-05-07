"""model_2.pyをimport"""
import matplotlib.pyplot as plt
from model_2 import ProdPlan


def main(nums, indpath, outputpath):
    """
    最適化問題を実行し、結果を表示する関数
    """
    # データのパスの指定
    ind_path = indpath
    # 出力先の指定
    output_path = outputpath
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
    prodplan.visualize(output_path)
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
    num_set = range(6, 14)

    for i in num_set:
        status, object_value, time, job_order = main(
            i, "./data/", f"./result/model_2/job_{i}"
        )
        time_list.append(time)
        job_order_list.append(job_order)

    # resultに計算時間のグラフを保存する
    plt.plot(num_set, time_list)
    plt.xlabel("Number of jobs")
    plt.ylabel("Time [s]")
    plt.savefig("./result/model_2/time.pdf", format="pdf")
    # resultに計算時間を保存する
    with open("./result/model_2/time.csv", "w") as f:
        for i in range(len(num_set)):
            f.write(str(num_set[i]) + "," + str(time_list[i]) + "\n")
    # resultにジョブの順序を保存する
    with open("./result/model_2/job_order.csv", "w") as f:
        for i in range(len(num_set)):
            f.write("model" + str(num_set[i]) + "," + str(job_order_list[i]) + "\n")
