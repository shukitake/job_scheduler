"""
実験用のファイル

model_1,2,3を実行して、結果を比較する。
ジョブ数は13とする。
"""
"""opt_plan_1.pyのmain関数をimport
   opt_plan_2.pyのmain関数をimport
   opt_plan_3.pyのmain関数をimport"""

import opt_plan_1 as opt_1
import opt_plan_2 as opt_2
import opt_plan_3 as opt_3

i = 12
status, object_value, time, job_order = opt_1.main(
    i, "./data/", f"./result/model_1/job_{i}/"
)
status, object_value, time, job_order = opt_2.main(
    i, "./data/", f"./result/model_1/job_{i}/"
)
status, object_value, time, job_order = opt_3.main(
    i, "./data/", f"./result/model_1/job_{i}/"
)
