import math
import random
import numpy as np
import pandas as pd
import itertools

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
             7591: 5,  7607: 40, 7621: 65, 7627: 9,  7630: 31,
             7643: 34, 7650: 13, 7661: 83, 7690: 6,  7747: 66,
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

working_month = {"OP" : [3, 4, 5, 6, 7, 8],                   "CZH": [9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7],
                 "BTH": [9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7], "TS" : [3, 4, 5, 6, 7, 8, 9, 10, 11],
                 "BTP": [2, 3, 4, 5, 6, 7, 8],                "BDP": [2, 3, 4, 5, 6, 7, 8],
                 "BDH": [9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7], "OH" : [9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7],
                 "MA" : [4, 5, 6, 7, 8, 9, 10, 11]}

lr = LinearRegression()


def read_in_Y(crop, consider_part):
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


def X_devide_region(consider_X):
    for p in provinces:
        temp_p_data = data[data["province"].map(lambda x: x == p)]
        for y in years:
            temp_py_data = temp_p_data[temp_p_data["year"].map(lambda x: x == y)]

            if temp_py_data[consider_X[0]].tolist():
                for x in consider_X:
                    name = str(p) + "_" + str(y)
                    if name not in X_region_year:
                        X_region_year[name] = [temp_py_data[x].tolist()]
                    else:
                        X_region_year[name].append(temp_py_data[x].tolist())


def normalize_X():
    for i in X_region_year:
        X_region_year_normalized[i] = []
        for j in X_region_year[i]:
            temp_array = np.array(j)
            X_region_year_normalized[i].append(((temp_array - temp_array.min()) / (temp_array.max() - temp_array.min())).tolist())


def init_list(crop, province):
    X = []
    Y = []
    for y in years:
        name = str(province) + "_" + str(y)
        if name in X_region_year_normalized and crop + "_" + name in crop_Y_year:
            X.append([np.average(X_region_year_normalized[name][i]) for i in range(len(X_region_year_normalized[name]))])
            Y.append(crop_Y_year[crop + "_" + name])
    return np.array(X), np.array(Y)


def add_power(data_X_array, powers):
    data_X_power_array = []

    for i in range(len(data_X_array)):
        data_X_power_array.append([])

        for j in range(len(data_X_array[i])):
            temp_list = []
            for p in range(1, powers[j] + 1):
                temp_list.append(data_X_array[i][j] ** p)
            data_X_power_array[-1].append(temp_list)

    return data_X_power_array


def predict_zero(data_Y_array):
    zero = data_Y_array.mean()

    RMSE = math.sqrt(((zero - data_Y_array) ** 2).sum() / len(data_Y_array))
    rRMSE = RMSE / data_Y_array.mean()

    return rRMSE


def predict(X_released, Y, divide_number):
    lr.fit(X_released, Y)

    coef = lr.coef_
    coef = coef.tolist()
    coef_divided = []
    degree_position = 0
    for i in range(len(divide_number)):
        coef_divided.append(coef[degree_position : degree_position + divide_number[i]])
        degree_position += divide_number[i]
    intercept = lr.intercept_

    y_predict = lr.predict(X_released)

    RMSE = math.sqrt(((y_predict - Y) ** 2).sum() / len(Y))
    rRMSE = RMSE / Y.mean()

    return rRMSE, coef, coef_divided, intercept


def distance(coefs_0, inter_0, X_nd, Y, original_rRMSE):
    new_predict = np.array([inter_0 for _ in X_nd])
    for i in range(len(X_nd)):
        for j in range(len(X_nd[0])):
            new_predict[i] += coefs_0[j] * X_nd[i][j]

    new_RMSE = math.sqrt(((new_predict - Y) ** 2).sum() / len(Y))
    new_rRMSE = new_RMSE / Y.mean()

    return math.fabs(new_rRMSE - original_rRMSE)


powers = [6, 6]
power_list = [[i for i in range(1, powers[j])] for j in range(len(powers))]
power_list = list(itertools.product(*power_list))

all_provinces = []
X_region_year = {}
X_region_year_normalized = {}
X_year_normalized_average_powered = {}
crops_Y_year = {}
Y_province = {}
rRMSE_degree = {}
coeffs_degree = {}
coeffs_degree_divided = {}
intercepts_degree = {}

X_devide_region(["rr24", "DJ_0"])
normalize_X()

for crop in crops:
    crop_Y_year = {}
    read_in_Y(crop, "rdt")
    crops_Y_year[crop] = crop_Y_year

    for p in provinces:
        X, Y = init_list(crop, p)

        if len(X) and len(Y):
            Y_province[crop + "_" + str(p)] = Y
            name = crop + "_" + str(p) + "_"
            rRMSE_degree[name + ('0_' * len(powers)).strip('_')] = predict_zero(Y)

            if p not in all_provinces:
                all_provinces.append(p)

            for pl in power_list:
                X_powered = add_power(X, pl)
                pl_name = str(pl)[1: -1].replace(' ', '').replace(',', '_').strip('_')

                X_released = []
                for i in X_powered:
                    X_released.append([])
                    for j in i:
                        X_released[-1] += j
                X_year_normalized_average_powered[str(p) + "_" + pl_name] = X_released

                rRMSE_degree[name + pl_name], coeffs_degree[name + pl_name], coeffs_degree_divided[name + pl_name], \
                intercepts_degree[name + pl_name] = predict(X_released, Y, pl)


crop = "OP"
pl = (1, 1)

cluster_number = 5
times = 10
max_time = 100

new_groups = []

for time in range(times):
    pl_name = str(pl)[1: -1].replace(' ', '').replace(',', '_').strip("_")
    random.shuffle(all_provinces)
    length = len(all_provinces) // cluster_number
    all_provinces_divided = [all_provinces[length * i : length * (i + 1)] for i in range(cluster_number)]
    length = sum([len(i) for i in all_provinces_divided])
    if length != len(all_provinces):
        for i in range(length, len(all_provinces)):
            all_provinces_divided[i - length].append(all_provinces[length])

    new_group = []
    for t in range(max_time):

        coef_group = {}
        coef_divided_group = {}
        inter_group = {}
        rRMSE_group = {}

        for i in range(cluster_number):
            X_group = []
            Y_group = []
            for j in all_provinces_divided[i]:
                X_group.append(X_year_normalized_average_powered[str(j) + "_" + pl_name])
                Y_group.append(Y_province[crop + "_" + str(j)])

            X_group_list = []
            Y_group_list = []
            for j in range(len(X_group)):
                X_group_list += X_group[j]
                Y_group_list += Y_group[j].tolist()

            rRMSE_group[i], coef_group[i], coef_divided_group[i], inter_group[i] = predict(X_group_list, np.array(Y_group_list), pl)

        province_point_distance = {}
        for p in provinces:
            for c in range(cluster_number):
                X_nd = X_year_normalized_average_powered[str(p) + "_" + pl_name]
                Y = Y_province[crop + "_" + str(p)]
                original_rRMSE = rRMSE_degree[crop + "_" + str(p) + "_" + pl_name]

                province_point_distance[str(p) + "_" + str(c)] = distance(coef_group[c], inter_group[c], X_nd, Y, original_rRMSE)

        new_group = [[] for i in range(cluster_number)]
        for p in all_provinces:
            temp_list = []
            for i in range(cluster_number):
                temp_list.append(province_point_distance[str(p) + "_" + str(i)])
            new_group[temp_list.index(min(temp_list))].append(p)

        if all_provinces_divided == new_group:
            break
        else:
            all_provinces_divided = new_group

    new_groups.append(new_group)


together_provinces = {}
for g in new_groups:
    for g0 in g:
        for p in range(len(g0)):
            for p0 in range(p + 1, len(g0)):
                if g0[p] in together_provinces:
                    together_provinces[g0[p]].append(g0[p0])
                else:
                    together_provinces[g0[p]] = [g0[p0]]
                if g0[p0] in together_provinces:
                    together_provinces[g0[p0]].append(g0[p])
                else:
                    together_provinces[g0[p0]] = [g0[p]]

together_provinces_number = {}
for i in together_provinces:
    temp_set = set(together_provinces[i])
    together_provinces_number[i] = {}
    for j in temp_set:
        together_provinces_number[i][j] = together_provinces[i].count(j)

threshold = 0.5

threshold = threshold * times + 0.5
real_together_province = {}
for i in together_provinces_number:
    for j in together_provinces_number[i]:
        if together_provinces_number[i][j] > threshold:
            if i in real_together_province:
                real_together_province[i].append(j)
            else:
                real_together_province[i] = [j]

clustered_province = []
for i in real_together_province:
    real_together_province[i].append(i)
    real_together_province[i].sort()

print("end")
