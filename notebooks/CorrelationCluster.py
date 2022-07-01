import math
import numpy as np
import pandas as pd

from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


data = pd.read_parquet("data/raw/france.parquet")

data.dropna(axis=0, how='any', inplace=True)

data["id"] = [i for i in range(len(data))]
data["time"] = data.index
data = data.set_index("id")
all_years = []
all_month = []
for t in range(len(data)):
    all_years.append(data["time"][t].year)
    all_month.append(data["time"][t].month)
data["year"] = all_years
data["month"] = all_month
data = data.drop(["time"], axis=1)

position = pd.read_csv("data/raw/postesSynop.csv", sep=";")

Id = position["ID"].astype(str)
for i in range(len(Id)):
    if len(Id[i]) < 5:
        Id[i] = '0' + Id[i]

production = pd.read_parquet("data/raw/franceagrimer-rdts-surfs-multicrops.parquet")

production = production.drop(production[production["n_dep"] == "2A"].index)
production = production.drop(production[production["n_dep"] == "2B"].index)
production = production.drop(production[production["n_dep"].astype(int) > 95].index)

provinces = {7005: 80, 7015: 59, 7020: 50, 7027: 14, 7037: 76,
             7072: 51, 7110: 29, 7117: 22, 7130: 35, 7139: 61,
             7149: 91, 7168: 10, 7181: 54, 7190: 67, 7207: 56,
             7222: 44, 7240: 37, 7255: 18, 7280: 21, 7299: 68,
             7314: 17, 7335: 86, 7434: 87, 7460: 63, 7471: 43,
             7481: 69, 7510: 33, 7535: 46, 7558: 12, 7577: 26,
             7591: 5, 7607: 40, 7621: 65, 7627: 9, 7630: 31,
             7643: 34, 7650: 13, 7661: 83, 7690: 6, 7747: 66,
             7761: 91, 67005: 10}

stations = data["id_sta"].unique()
unwanted_stations = []
for i in stations:
    if i not in provinces:
        unwanted_stations.append(i)
for i in unwanted_stations:
    data = data.drop(data[data["id_sta"] == i].index)

temp_province = []
for i in data["id_sta"]:
    temp_province.append(provinces[i])
data["province"] = temp_province
data = data.drop(["id_sta"], axis=1)

years = data["year"].unique()
provinces = data["province"].unique()
crops = production["crop"].unique()
n_deps = production["n_dep"].unique()

working_month = {"OP": [3, 4, 5, 6, 7, 8], "CZH": [9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7],
                 "BTH": [9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7], "TS": [3, 4, 5, 6, 7, 8, 9, 10, 11],
                 "BTP": [2, 3, 4, 5, 6, 7, 8], "BDP": [2, 3, 4, 5, 6, 7, 8],
                 "BDH": [9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7], "OH": [9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7],
                 "MA": [4, 5, 6, 7, 8, 9, 10, 11]}

lr = LinearRegression()


def read_in_Y(crop, consider_part):
    crop_Y_year = {}

    map_crop = production['crop'].map(lambda x: x == crop)
    crop_value = production[map_crop]

    for n in n_deps:
        map_province = crop_value['n_dep'].map(lambda x: x == n)
        crop_n_value = crop_value[map_province]

        for y in years:
            if len(crop_n_value[consider_part + "_" + str(y)].values):
                rdt_value = crop_n_value[consider_part + "_" + str(y)].values[0]

                if rdt_value:
                    crop_Y_year[crop + "_" + str(int(n)) + "_" + str(y)] = rdt_value

    return crop_Y_year


def read_in_Ys():
    for crop in crops:
        if crop not in crops_Y_year:
            crops_Y_year[crop] = read_in_Y(crop, "rdt")


def X_devide_region(consider_X):
    for p in provinces:
        temp_p_data = data[data["province"].map(lambda x: x == p)]

        for y in years:
            temp_py_data = temp_p_data[temp_p_data["year"].map(lambda x: x == y)]

            for m in range(1, 13):
                temp_pym_data = temp_py_data[temp_py_data["month"].map(lambda x: x == m)]

                if consider_X[0] in temp_pym_data and temp_pym_data[consider_X[0]].tolist():
                    for x in consider_X:
                        name = str(p) + "_" + str(y) + "_" + str(m)
                        if name not in X_region_year_month:
                            X_region_year_month[name] = [temp_pym_data[x].tolist()]
                        else:
                            X_region_year_month[name].append(temp_pym_data[x].tolist())


def normalize_X():
    for i in X_region_year_month:
        X_region_year_month_normalized[i] = []

        for j in X_region_year_month[i]:
            temp_array = np.array(j)
            max_X, min_X = max(temp_array), min(temp_array)

            if max_X - min_X > 1.0e-15:
                X_region_year_month_normalized[i].append(((temp_array - min_X) / (max_X - min_X)).tolist())
            else:
                X_region_year_month_normalized[i].append([len(j) - 1 for _ in j])


def average_X():
    for i in X_region_year_month_normalized:
        X_region_year_normalized_average[i] = []
        for j in range(len(X_region_year_month_normalized[i])):
            X_region_year_normalized_average[i].append(np.average(X_region_year_month_normalized[i][j]))


def init_list(crop, consider_part, province, month):
    X = []
    Y = []
    crop_Y_year = crops_Y_year[crop]

    if consider_part not in X_consider_part:
        temp_X = {}
        for i in X_region_year_normalized_average:
            temp_X[i] = X_region_year_normalized_average[i][consider_part]
        X_consider_part[consider_part] = temp_X
    else:
        temp_X = X_consider_part[consider_part]

    for i in temp_X:
        p, y, m = i.split("_")
        if int(p) == province and int(m) == month:
            name = crop + "_" + p + "_" + y
            if name in crop_Y_year:
                X.append(temp_X[i])
                Y.append(crop_Y_year[name])

    return np.array(X), np.array(Y)


def add_power(X, powers):
    powered_X = []

    for i in range(len(X)):
        powered_X.append([])

        for j in range(len(X[i])):
            temp_list = []
            for p in range(1, powers[j] + 1):
                temp_list.append(X[i][j] ** p)
            for t in temp_list:
                powered_X[-1].append(t)

    return powered_X


def correlation(X, Y):
    if len(X) < 3 or len(Y) < 3:
        return 0.0

    avg_X = np.average(X)
    avg_Y = np.average(Y)

    cr = 0
    nx = 0
    ny = 0
    for i in range(len(X)):
        cr += (X[i] - avg_X) * (Y[i] - avg_Y)
        nx += (X[i] - avg_X) ** 2
        ny += (Y[i] - avg_Y) ** 2

    if math.sqrt(nx) * math.sqrt(ny) > 1.0e-15:
        r = cr / (math.sqrt(nx) * math.sqrt(ny))
    else:
        r = 0.0

    return r


def predict_zero(data_Y_array):
    zero = data_Y_array.mean()

    RMSE = math.sqrt(((zero - data_Y_array) ** 2).sum() / len(data_Y_array))
    rRMSE = RMSE / data_Y_array.mean()

    return rRMSE


def predict_XY(times, X, Y):
    sum_RMSE = 0
    coef = np.array([0.0 for i in range(len(X[0]))])
    intercept = 0.0

    for i in range(times):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

        lr.fit(X_train, y_train)

        coef += lr.coef_
        intercept += lr.intercept_

        y_predict = lr.predict(X_test)

        RMSE = math.sqrt(((y_predict - y_test) ** 2).sum() / len(y_test))
        rRMSE = RMSE / y_test.mean()

        sum_RMSE += rRMSE

    return sum_RMSE / times, coef / times, intercept / times


def get_correltion():
    for p in provinces:
        consider_parts_list = []
        correlations = {}
        correlations_list = []

        for cp in range(len(consider_parts)):
            for m in range(1, 13):
                correlations_list.append([])
                consider_parts_list.append(consider_parts[cp] + "_" + str(m))

                for crop in crops:
                    X, Y = init_list(crop, cp, p, m)
                    correlations[crop + "_" + consider_parts[cp] + "_" + str(m)] = correlation(X, Y)
                    correlations_list[-1].append(correlations[crop + "_" + consider_parts[cp] + "_" + str(m)])

        c_plot = pd.DataFrame(data=correlations_list, columns=crops)
        c_plot["type"] = consider_parts_list
        c_plot = c_plot.set_index("type")
        correlation_province[p] = c_plot


def treat_noise():
    provinces = list(correlation_province.keys())

    for c in consider_parts:
        for crop in crops:
            for p in provinces:
                temp_list = [correlation_province[p][crop][c + "_" + str(i)] for i in range(1, 13)]
                temp_list.sort()
                for i in range(max_hold):
                    if math.fabs(temp_list[0]) > math.fabs(temp_list[-1]):
                        temp_list.pop(0)
                    else:
                        temp_list.pop(-1)

                for i in range(1, 13):
                    temp_c = correlation_province[p][crop][c + "_" + str(i)]

                    if temp_c in temp_list:
                        temp_list.remove(temp_c)
                        correlation_province[p][crop][c + "_" + str(i)] = 0.0


def do_cluster(cut_number=0.2):
    temp_clustering_result = {}

    for crop in crops:
        X = []

        X_provinces = list(provinces.copy())
        for p in provinces:
            X.append([])

            for c in consider_parts:
                for i in range(1, 13):
                    X[-1].append(correlation_province[p][crop][c + "_" + str(i)])

            X_sort = X[-1].copy()
            X_sort.sort()
            if math.fabs(X_sort[0]) < 1.0e-15 and math.fabs(X_sort[-1]) < 1.0e-15:
                X.pop(-1)
                X_provinces.remove(p)

        crop_X_provinces[crop] = X_provinces.copy()

        for nc in n_clusters_apply:
            if str(len(X)) + "_" + str(nc) not in var_limits:
                global_var(len(X), nc, cut_number)

            for aa in affinity_apply:
                for la in linkage_apply:
                    if la == "ward" and aa != "euclidean":
                        continue
                    if aa == "cosine" and [0.0 for _ in range(len(X[0]))] in X:
                        continue

                    clustering = AgglomerativeClustering(n_clusters=nc, affinity=aa, linkage=la).fit(X)
                    var_counts = [list(clustering.labels_).count(i) for i in range(nc)]
                    var_counts = math.sqrt(np.var(var_counts))

                    if var_counts < var_limits[str(len(X)) + "_" + str(nc)]:
                        temp_clustering_result[crop + "_" + str(nc) + "_" + aa + "_" + la] = clustering.labels_

        for i in temp_clustering_result:
            crop, nc, aa, la = i.split("_")
            province_cluster_group[i] = [[] for _ in range(int(nc))]
            for p in range(len(X_provinces)):
                province_cluster_group[i][temp_clustering_result[i][p]].append(X_provinces[p])

        clustering_result.update(temp_clustering_result)
        temp_clustering_result.clear()


def evaluate_clusters():
    for cr in province_cluster_group:
        crop, nc, aa, la = cr.split("_")

        n = 0
        for pcg in province_cluster_group[cr]:
            X = []
            Y = []

            for p in pcg:
                for m in range(1, 13):
                    for c in consider_parts:
                        temp_X, temp_Y = init_list(crop, consider_parts.index(c), p, m)
                        temp_X, temp_Y = list(temp_X), list(temp_Y)
                        X += temp_X
                        Y += temp_Y

            all_amount[cr + "_" + str(n)] = len(Y)

            cluster_evaluate_result[cr + "_" + str(n)] = [predict_zero(np.array(Y)) * len(Y)]

            X = np.array(X).reshape(-1, 1)
            Y = np.array(Y)
            for d in degrees:
                powered_X = add_power(X, [d])
                rRMSE = predict_XY(times, powered_X, Y)[0]
                cluster_evaluate_result[cr + "_" + str(n)].append(rRMSE * len(Y))

            n += 1

        all_cluster_rRMSE[cr] = 0.0
        temp_number = sum(list(all_amount.values()))

        for i in range(int(nc)):
            temp_list = cluster_evaluate_result[cr + "_" + str(i)]
            all_cluster_rRMSE[cr] += min(temp_list) / temp_number

        cluster_evaluate_result.clear()
        all_amount.clear()


def global_var(length, type_number, cut_number=0.2):
    if math.factorial(length - 1) / (math.factorial(type_number - 1) * math.factorial(length - type_number)) > 1.0e+6:
        for i in range(2, type_number):
            if str(length) + "_" + str(i) not in var_limits:
                global_var(length, i, cut_number)

        percents = np.average([var_limits[str(length) + "_" + str(i + 1)] / var_limits[str(length) + "_" + str(i)] for i in range(2, type_number - 1)])
        var_limits[str(length) + "_" + str(type_number)] = var_limits[str(length) + "_" + str(type_number - 1)] * percents
        return

    temp_var_limits = {}

    for tn in range(2, type_number + 1):
        if str(length) + "_" + str(tn) in var_limits:
            continue

        dist_list = [[i] for i in range(1, length)]
        next_dist_list = []

        for t in range(tn - 2):
            for i in dist_list:
                for j in range(1, length + 1):
                    if sum(i) + j < length:
                        next_dist_list.append(i + [j])
                    else:
                        break

            dist_list = next_dist_list
            next_dist_list = []

        for i in dist_list:
            i.append(length - sum(i))

        temp_var_limits[tn] = [math.sqrt(np.var(k)) for k in dist_list]
        temp_var_limits[tn].sort()

    for vl in temp_var_limits:
        count = 0
        for v in temp_var_limits[vl]:
            if v > cut_number * temp_var_limits[vl][-1]:
                count += 1
        var_limits[str(length) + "_" + str(vl)] = temp_var_limits[vl][count]


def get_best_in_degree():
    for cr in all_cluster_rRMSE:
        crop, nc, aa, la = cr.split("_")

        name = crop + "_" + str(nc)
        if name not in best_cluster_rRMSE or all_cluster_rRMSE[cr] < best_cluster_rRMSE[name]:
            best_cluster_rRMSE[name] = all_cluster_rRMSE[cr]
            best_cluster_param[name] = aa + "_" + la


def get_best_model_no_over(cut_number=0.2):
    for crop in crops:
        temp_list = []
        temp_list_index = []

        for nca in n_clusters_apply:
            if crop + "_" + str(nca) in best_cluster_rRMSE:
                temp_list.append(best_cluster_rRMSE[crop + "_" + str(nca)])
                temp_list_index.append(nca)

        temp_diff = temp_list[0] - temp_list[1]
        for i in range(1, len(temp_list) - 1):
            d = temp_list[i] - temp_list[i + 1]

            if d < 0 or d / (temp_diff + d) < cut_number:
                best_model_not_overmodel[crop] = temp_list_index[i]
                break
            elif i == len(temp_list) - 2:
                best_model_not_overmodel[crop] = -1
            else:
                temp_diff += d


# READ IN

consider_parts = ["rr24", "t_avg"]

X_region_year_month = {}
X_region_year_month_normalized = {}
X_region_year_normalized_average = {}
crops_Y_year = {}
X_consider_part = {}

X_devide_region(consider_parts)
normalize_X()
average_X()
read_in_Ys()

del X_region_year_month
del X_region_year_month_normalized


# CORRELATION CALCULATE

correlation_province = {}

get_correltion()


# NOISE TREAT

max_hold = 5

treat_noise()


# CLUSTER WITH AGGLOMERATIVE

n_clusters_apply = [2, 3, 4, 5, 6, 7, 8, 9, 10]

affinity_apply = ["euclidean", "l1", "l2", "manhattan", "cosine"]
linkage_apply = ["ward", "complete", "average", "single"]
clustering_result = {}
var_limits = {}
province_cluster_group = {}
crop_X_provinces = {}

do_cluster()


# EVALUATE

degrees = [1, 2, 3, 4, 5, 6]
times = 100

cluster_evaluate_result = {}
all_cluster_rRMSE = {}
all_amount = {}

evaluate_clusters()


# GET BEST AMONG DEGREE

best_cluster_rRMSE = {}
best_cluster_param = {}

get_best_in_degree()


# GET BEST MODEL NOT OVERMODEL

best_model_not_overmodel = {}

get_best_model_no_over()


# PRINT RESULT

for i in best_model_not_overmodel:
    if best_model_not_overmodel[i] != -1:
        print(i + ": " + str(best_model_not_overmodel[i]) + " parts")
        print("Method: " + str(best_cluster_param[i + "_" + str(best_model_not_overmodel[i])]))

        name = i + "_" + str(best_model_not_overmodel[i])
        name = name + "_" + best_cluster_param[name]

        best_cluster_result = clustering_result[name]
        province_cluster_groups = [[] for j in range(best_model_not_overmodel[i])]

        provinces = crop_X_provinces[i]
        for p in range(len(provinces)):
            province_cluster_groups[best_cluster_result[p]].append(provinces[p])
        for j in range(len(province_cluster_groups)):
            print("Part " + str(j) + ": " + str(province_cluster_groups[j])[1: -1])

        print("rRMSE: " + str(best_cluster_rRMSE[i + "_" + str(best_model_not_overmodel[i])]))

        print()


print("End")
