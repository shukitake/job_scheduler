import csv

import model.model3
import model.model_1
import model.model_2

with open("result.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(
        ["model", "J", "status", "value", "time", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    )

    for i in range(1, 11):
        status, value, t, sorted_job = model_1.main(i)
        writer.writerow(["定式化1", i, status, value, t] + sorted_job)

        status, value, t, sorted_job = model_2.main(i)
        writer.writerow(["定式化2", i, status, value, t] + sorted_job)

        status, value, t, sorted_job = model_3.main(i)
        writer.writerow(["定式化3", i, status, value, t] + sorted_job)
