import copy
import os
import time
import graph
import openpyxl
import pulp
import numpy as np
from scipy.stats import norm
import instance
import sympy
from scipy import optimize
# 手術クラス

home = os.environ['HOME']
directory_path = home + '/Documents/data-july'
path = directory_path + '/operations_with_jisseki_remake.csv'
surgeon_info_path = directory_path + '/surgeon_info.csv'
surgeries_path = directory_path + '/surgeries.csv'
distribution_path = home + '/Documents/surgerySchedule/distribution.csv'
num_surgery = 10
num_date = 1
seed = 1
alpha = 0.95
surgery_instance = instance.SurgeryInstance(path, surgeon_info_path, num_surgery, num_date, seed, surgeries_path, distribution_path)
list_room, list_surgery, list_surgeon, list_date, G = surgery_instance.get_sets()
distribution_dict = surgery_instance.get_dict_distribution()
dict_surgery_id_room = surgery_instance.get_room_surgery_dict()

print('手術件数:{:}'.format(len(list_surgery)))
print('外科医の人数:{:}'.format(len(list_surgeon)))
url = home + '/Documents/solution/solution.xlsx'
url2 = home + '/Documents/solution/solution2.xlsx'

dict_not_available_room = {}
for surgery in list_surgery:
    surgery_type_list = surgery.get_surgery_type()
    available_room = set(list_room)
    for surgery_type in surgery_type_list:
        if surgery_type in dict_surgery_id_room.keys():
            available_room = available_room & set(dict_surgery_id_room[surgery_type])
    not_available_room = set(list_room) - available_room
    dict_not_available_room[surgery] = list(not_available_room)

def aprox_norm_cdf(x):
    return 1 - 1 / np.sqrt(2 * np.pi) * sympy.exp(-x**2 / 2) / (0.226 + 0.64 * x + 0.33 * sympy.sqrt(x**2 + 3))

def log_normal(x, mu, sigma):
    return 1 / (np.sqrt(2 * np.pi * sigma) * x) * np.exp(-(np.log(x) - mu) ** 2 / (2 * sigma))

def diff_log_normal(x, mu, sigma):
    return 1 / (np.sqrt(2 * np.pi * sigma) * x ** 2) * np.exp(-(np.log(x) - mu) ** 2 / (2 * sigma)) * (1 + (np.log(x) - mu) / sigma)

def log_cumulative(x, mu, sigma):
    return np.log(norm.cdf((np.log(x) - mu) / np.sqrt(sigma)))
def first_order(x, mu, sigma):
    return log_normal(x, mu, sigma) / norm.cdf((np.log(x) - mu) / np.sqrt(sigma))

def second_order(x, mu, sigma):
    return (norm.cdf((np.log(x) - mu) / np.sqrt(sigma)) * diff_log_normal(x, mu, sigma) - log_normal(x, mu, sigma) ** 2) / (norm.cdf((np.log(x) - mu) / np.sqrt(sigma))) ** 2

def aux_obj(x, mu, sigma, pi, pi_alpha):
    return pi * x - pi_alpha * log_cumulative(x, mu, sigma)

def newton_method(x, mu, sigma, pi, pi_alpha):
    eps = 10 ** -3
    while np.abs(-pi - pi_alpha * first_order(x, mu, sigma)) > eps:
        print('error={:}'.format(np.abs(-pi - pi_alpha * first_order(x, mu, sigma))))
        x = x - 0.1 * first_order(x, mu, sigma) / second_order(x, mu, sigma)
    return x

def get_sigma(mean, variance):
    return np.log(variance / (mean ** 2) + 1)

def get_mu(mean, variance):
    return np.log(mean) - get_sigma(mean, variance) / 2


def get_right_hand(alpha, mean, variance):
    sigma = get_sigma(mean, variance)
    mu = get_mu(mean, variance)
    return np.exp(sigma * norm.ppf(alpha) + mu)


num_int_var = 0
num_con_var = 0
num_constraint = 0

initial = time.time()

T = 480
PT = 15
M = 99999
ope_time = 9
O_max = 120

waiting_list = []

solver = pulp.CPLEX_CMD(msg=0)

def to_time(num):
    hour = int(num / 60)
    minute = int(num % 60)
    if hour < 10:
        hour = "0" + str(hour)
    if minute < 10:
        minute = "0" + str(minute)
    return str(hour) + ":" + str(minute)


def to_time_of_day(num):
    hour = int(num / 60) + 9
    minute = int(num % 60)
    if hour < 10:
        hour = "0" + str(hour)
    if minute < 10:
        minute = "0" + str(minute)
    return str(hour) + ":" + str(minute)


def get_total_time(surgery):
    return surgery.get_preparation_time() + surgery.get_surgery_time() + surgery.get_cleaning_time()


def get_entry_time(surgery):
    return ts_val[surgery] - surgery.get_preparation_time()


def get_end_time(surgery):
    return ts_val[surgery] + surgery.get_surgery_time()


def get_exit_time(surgery):
    return get_end_time(surgery) + surgery.get_cleaning_time()

num = 0
start = time.time()
list_surgery_copied = copy.copy(list_surgery)
L = []
x_val = {}
for surgery in list_surgery:
    for r in list_room:
        for d in list_date:
            x_val[surgery, r, d] = 0
ot_val = {}
for r in list_room:
    for d in list_date:
        ot_val[r, d] = 0

n_val = {}
for surgeon in list_surgeon:
    for d in list_date:
        n_val[surgeon, d] = 0

wt_val = {}
for surgeon in list_surgeon:
    for d in list_date:
        wt_val[surgeon, d] = 0

ts_val = {}
for surgery in list_surgery:
    ts_val[surgery] = 0

x = {}
q = {}
y = {}
z = {}
n = {}
ts = {}
msR = {}
ot = {}
tsS = {}
msS = {}
wt = {}
ost = {}
ns = {}

book = openpyxl.Workbook()
book2 = openpyxl.Workbook()
for d in list_date:
    print("day={:}".format(d))
    feasiblity_criteria = False
    if len(list_surgery_copied) == 0:
        break
    for surgery in list_surgery_copied[:]:
        if surgery.get_release_date() <= d:
            if surgery not in waiting_list:
                waiting_list.append(surgery)
                list_surgery_copied.remove(surgery)

    l = 0
    while not feasiblity_criteria:
        l += 1
        print('l={:}'.format(l))
        for surgery in waiting_list:
            for r in list_room:
                num_int_var += 1
                x[surgery, r, d, l] = pulp.LpVariable("x({:},{:},{:},{:})".format(surgery.get_surgery_id(), r, d, l), cat='Binary')

        for surgery in waiting_list:
            for surgeon in list_surgeon:
                num_int_var += 1
                q[surgery, surgeon, d, l] = pulp.LpVariable("q({:},{:},{:},{:})".format(surgery.get_surgery_id(), surgeon.get_surgeon_id(), d, l), cat='Binary')

        for surgeon in list_surgeon:
            num_int_var += 1
            n[surgeon, d, l] = pulp.LpVariable("n({:},{:})".format(surgeon.get_surgeon_id(), d, l), cat='Binary')

        for surgery in waiting_list:
            num_con_var += 1
            ts[surgery, l] = pulp.LpVariable("ts({:},{:})".format(surgery.get_surgery_id(), l), lowBound=0, cat='Continuous')

        for r in list_room:
            num_con_var += 1
            msR[r, d, l] = pulp.LpVariable("msR({:},{:},{:})".format(r, d, l), lowBound=0, cat='Continuous')

        for r in list_room:
            num_con_var += 1
            ot[r, d, l] = pulp.LpVariable("ot({:},{:},{:})".format(r, d, l), lowBound=0, cat='Continuous')

        for surgeon in list_surgeon:
            tsS[surgeon, d, l] = pulp.LpVariable("tsS({:},{:},{:})".format(surgeon.get_surgeon_id(), d, l), lowBound=0, cat='Continuous')

        for surgeon in list_surgeon:
            msS[surgeon, d, l] = pulp.LpVariable("msS({:},{:},{:})".format(surgeon.get_surgeon_id(), d, l), lowBound=0, cat='Continuous')

        for surgeon in list_surgeon:
            num_con_var += 1
            wt[surgeon, d, l] = pulp.LpVariable("wt({:},{:},{:})".format(surgeon.get_surgeon_id(), d, l), lowBound=0, cat='Continuous')

        operating_room_planning = pulp.LpProblem("ORP" + str(l), pulp.LpMinimize)
        objective = pulp.lpSum((((d - surgery.get_release_date()) + max(d - surgery.get_due_date(), 0)) * x[surgery, r, d, l] * surgery.get_priority()) for surgery in waiting_list for r in list_room)
        objective += 10000 * pulp.lpSum(((surgery.get_due_date() - surgery.get_release_date()) + (len(list_date) + 1 - surgery.get_due_date())) * surgery.get_priority() * (1 - pulp.lpSum(x[surgery, r, d, l] for r in list_room)) for surgery in waiting_list)
        objective += pulp.lpSum(ot[r, d, l] for r in list_room)
        objective += pulp.lpSum(q[surgery, surgeon, d, l] * 1 / (float(surgeon.get_dict_group()[surgery.get_group()])) for surgery in waiting_list for surgeon in list_surgeon if float(surgeon.get_dict_group()[surgery.get_group()]) > 0)
        operating_room_planning += objective
        for surgery in waiting_list:
            operating_room_planning += pulp.lpSum(x[surgery, r, d, l] for r in list_room) <= 1
        for surgery in waiting_list:
            operating_room_planning += pulp.lpSum(q[surgery, surgeon, d, l] for surgeon in list_surgeon) <= 1

        for surgery in waiting_list:
            operating_room_planning += pulp.lpSum(x[surgery, r, d, l] for r in list_room) == pulp.lpSum(q[surgery, surgeon, d, l] for surgeon in list_surgeon)
        """
        for surgeon in list_surgeon:
            operating_room_planning += pulp.lpSum(q[surgery, surgeon.get_surgeon_id(), d, l] for surgery in waiting_list) <= len(waiting_list) * n[surgeon, d, l]

        for surgeon in list_surgeon:
            operating_room_planning += n[surgeon, d, l] <= pulp.lpSum(q[surgery, surgeon.get_surgeon_id(), d, l] for surgery in waiting_list)

        for surgeon in list_surgeon:
            operating_room_planning += pulp.lpSum(q[surgery, surgeon.get_surgeon_id(), d, l] for surgery in waiting_list) <= 3
        """
        for r in list_room:
            operating_room_planning += pulp.lpSum((surgery.get_preparation_time() + surgery.get_surgery_time() + surgery.get_cleaning_time()) * x[surgery, r, d, l] for surgery in waiting_list) - T <= ot[r, d, l]

        for surgery in waiting_list:
            for r in dict_not_available_room[surgery]:
                operating_room_planning += x[surgery, r, d, l] == 0

        for surgery in waiting_list:
            group = surgery.get_group()
            for surgeon in list_surgeon:
                if float(surgeon.get_dict_group()[group]) <= 0:
                    operating_room_planning += q[surgery, surgeon, d, l] == 0
        if l > 1:
            for i in range(1, l):
                for surgery1 in waiting_list:
                    for surgery2 in waiting_list:
                        if surgery1 != surgery2:
                            operating_room_planning += pulp.lpSum(x[surgery1, r, d, l] for r in list_room if x[surgery1, r, d, i].value() == 1) + pulp.lpSum(x[surgery2, r, d, l] for r in list_room if x[surgery2, r, d, i].value() == 1) + pulp.lpSum(q[surgery1, surgeon, d, l] for surgeon in list_surgeon if q[surgery1, surgeon, d, i].value() == 1) + pulp.lpSum(q[surgery2, surgeon, d, l] for surgeon in list_surgeon if q[surgery2, surgeon, d, i].value() == 1) <= 3

        result_status = operating_room_planning.solve(solver)
        print('ORP={:}'.format(pulp.LpStatus[result_status]))
        feasibility_criteria2 = False
        iterations = 0
        planned_surgery = []
        x_val = {}
        for surgery in waiting_list:
            for r in list_room:
                if round(x[surgery, r, d, l].value()) == 1:
                    x_val[surgery, r, d, l] = 1
                else:
                    x_val[surgery, r, d, l] = 0
        q_val = {}
        for surgery in waiting_list:
            for surgeon in list_surgeon:
                if round(q[surgery, surgeon, d, l].value()) == 1:
                    q_val[surgery, surgeon, d, l] = 1
                else:
                    q_val[surgery, surgeon, d, l] = 0

        for surgery in waiting_list:
            if sum(x_val[surgery, r, d, l] for r in list_room) == 1:
                planned_surgery.append(surgery)
        planned_surgeon = []
        for surgeon in list_surgeon:
            if sum(q_val[surgery, surgeon, d, l] for surgery in waiting_list) >= 1:
                planned_surgeon.append(surgeon)

        dict_surgery_room = {}
        for r in list_room:
            list_surgery_in_r = [surgery for surgery in planned_surgery if x_val[surgery, r, d, l] == 1]
            dict_surgery_room[r] = list_surgery_in_r
        dict_surgery_surgeon = {}
        for surgeon in planned_surgeon:
            list_surgery_by_k = [surgery for surgery in planned_surgery if q_val[surgery, surgeon, d, l] == 1]
            dict_surgery_surgeon[surgeon] = list_surgery_by_k

        omega1 = {}
        mu1 = {}
        sigma1 = {}
        omega1_length = len(planned_surgery)
        for surgery in planned_surgery:
            sigma = distribution_dict[surgery.get_group(), 'preparation_sigma']
            mu = distribution_dict[surgery.get_group(), 'preparation_mu']
            mu1[surgery] = mu
            sigma1[surgery] = sigma
            omega1[surgery, 1] = np.exp(norm.ppf(alpha ** (1 / omega1_length)) * np.sqrt(sigma) + mu)


        omega2 = {}
        mu2 = {}
        sigma2 = {}
        omega2_length = sum(len(dict_surgery_room[r]) * (len(dict_surgery_room[r]) - 1) for r in list_room)
        omega2_index_list = {}
        num = 0
        for r in list_room:
            surgery_in_r = dict_surgery_room[r]
            for surgery1 in surgery_in_r:
                for surgery2 in surgery_in_r:
                    if surgery1 != surgery2:
                        omega2_index_list[num] = [surgery1, surgery2]
                        num += 1
                        mean = distribution_dict[surgery1.get_group(), 'surgery_mean'] + distribution_dict[surgery1.get_group(), 'cleaning_mean'] + distribution_dict[surgery2.get_group(), 'preparation_mean']
                        variance = distribution_dict[surgery1.get_group(), 'surgery_variance'] + distribution_dict[surgery1.get_group(), 'cleaning_variance'] + distribution_dict[surgery2.get_group(), 'preparation_variance']
                        sigma = np.log(variance / (mean**2) + 1)
                        mu = np.log(mean) - sigma / 2
                        mu2[surgery1, surgery2] = mu
                        sigma2[surgery1, surgery2] = sigma
                        omega2[surgery1, surgery2, 1] = np.exp(norm.ppf(alpha ** (1 / omega2_length)) * np.sqrt(sigma) + mu)
        omega3_length = sum(len(dict_surgery_surgeon[surgeon]) * (len(dict_surgery_surgeon[surgeon]) - 1) for surgeon in planned_surgeon)
        omega3 = {}
        mu3 = {}
        sigma3 = {}
        omega3_index_list = {}
        num = 0
        for surgeon in planned_surgeon:
            surgery_by_k = dict_surgery_surgeon[surgeon]
            for surgery1 in surgery_by_k:
                for surgery2 in surgery_by_k:
                    if surgery1 != surgery2:
                        omega3_index_list[num] = [surgery1, surgery2]
                        num += 1
                        mu = distribution_dict[surgery1.get_group(), 'surgery_mu']
                        sigma = distribution_dict[surgery1.get_group(), 'surgery_sigma']
                        mu3[surgery1, surgery2] = mu
                        sigma3[surgery1, surgery2] = sigma
                        omega3[surgery1, surgery2, 1] = np.exp(norm.ppf(alpha ** (1 / omega3_length)) * np.sqrt(sigma) + mu)
        omega4 = {}
        mu4 = {}
        sigma4 = {}
        omega4_length = sum(len(dict_surgery_room[r]) for r in list_room)
        omega4_index_list = {}
        num = 0
        for r in list_room:
            surgery_in_r = dict_surgery_room[r]
            for surgery in surgery_in_r:
                omega4_index_list[num] = surgery
                num += 1
                mean = distribution_dict[surgery.get_group(), 'surgery_mean'] + distribution_dict[surgery.get_group(), 'cleaning_mean']
                variance = distribution_dict[surgery.get_group(), 'surgery_variance'] + distribution_dict[surgery.get_group(), 'cleaning_variance']
                sigma = np.log(variance / (mean**2) + 1)
                mu = np.log(mean) - sigma / 2
                mu4[surgery] = mu
                sigma4[surgery] = sigma
                omega4[surgery, 1] = np.exp(norm.ppf(alpha ** (1 / omega4_length)) * np.sqrt(sigma) + mu)
        print('num={:}'.format(num))
        print('omega4_length={:}'.format(omega4_length))
        print(len(omega4))
        omega5 = {}
        mu5 = {}
        sigma5 = {}
        omega5_length = sum(len(dict_surgery_surgeon[surgeon]) for surgeon in planned_surgeon)
        omega5_index_list = {}
        num = 0
        for surgeon in planned_surgeon:
            surgery_by_k = dict_surgery_surgeon[surgeon]
            for surgery in surgery_by_k:
                omega5_index_list[num] = surgery
                num += 1
                mu = distribution_dict[surgery.get_group(), 'surgery_mu']
                sigma = distribution_dict[surgery.get_group(), 'surgery_sigma']
                mu5[surgery] = mu
                sigma5[surgery] = sigma
                omega5[surgery, 1] = np.exp(norm.ppf(alpha ** (1 / omega5_length)) * np.sqrt(sigma) + mu)
        print('num={:}'.format(num))
        print('omega5_length={:}'.format(omega5_length))
        print(len(omega5))

        omega6 = {}
        mu6 = {}
        sigma6 = {}
        omega6_length = len(planned_surgeon)
        omega6_index_list = {}
        num = 0
        for surgeon in planned_surgeon:
            omega6_index_list[num] = surgeon
            num += 1
            mean = sum(distribution_dict[surgery.get_group(), 'surgery_mean'] for surgery in dict_surgery_surgeon[surgeon])
            variance = sum(distribution_dict[surgery.get_group(), 'surgery_variance'] for surgery in dict_surgery_surgeon[surgeon])
            sigma = np.log(variance / (mean**2) + 1)
            mu = np.log(mean) - sigma / 2
            mu6[surgeon] = mu
            sigma6[surgeon] = sigma
            omega6[surgeon, 1] = np.exp(norm.ppf(1 - alpha ** (1 / omega6_length)) * np.sqrt(sigma) + mu)
        print('num={:}'.format(num))
        print('omega6_length={:}'.format(omega6_length))
        print(len(omega6))

        stopping_criteria = np.inf
        while not feasibility_criteria2:
            iterations += 1
            print('iterations={:}'.format(iterations))
            for j in range(2):
                lam = {}
                for t in range(1, iterations + 1):
                    lam[t] = pulp.LpVariable('lam({:})'.format(t), lowBound=0, cat='Continuous')
                if j == 1:
                    for surgery1 in waiting_list:
                        for surgery2 in waiting_list:
                            if surgery1 != surgery2:
                                for r in list_room:
                                    y[surgery1, surgery2, r, d, l] = pulp.LpVariable("y({:},{:},{:},{:},{:})".format(surgery1.get_surgery_id(), surgery2.get_surgery_id(), r, d, l), cat='Binary')

                    for surgery1 in waiting_list:
                        for surgery2 in waiting_list:
                            if surgery1 != surgery2:
                                for surgeon in list_surgeon:
                                    z[surgery1, surgery2, surgeon, d, l] = pulp.LpVariable("z({:},{:},{:},{:},{:})".format(surgery1.get_surgery_id(), surgery2.get_surgery_id(), surgeon.get_surgeon_id(), d, l), cat='Binary')
                else:
                    for surgery1 in waiting_list:
                        for surgery2 in waiting_list:
                            if surgery1 != surgery2:
                                for r in list_room:
                                    y[surgery1, surgery2, r, d, l] = pulp.LpVariable("y({:},{:},{:},{:},{:})".format(surgery1.get_surgery_id(), surgery2.get_surgery_id(), r, d, l), lowBound=0, upBound=1, cat='Continuous')

                    for surgery1 in waiting_list:
                        for surgery2 in waiting_list:
                            if surgery1 != surgery2:
                                for surgeon in list_surgeon:
                                    z[surgery1, surgery2, surgeon, d, l] = pulp.LpVariable("z({:},{:},{:},{:},{:})".format(surgery1.get_surgery_id(), surgery2.get_surgery_id(), surgeon.get_surgeon_id(), d, l), lowBound=0, upBound=1, cat='Continuous')
                operating_room_scheduling = pulp.LpProblem('ORS', pulp.LpMinimize)
                operating_room_scheduling += pulp.lpSum(ot[r, d, l] for r in list_room) + pulp.lpSum(wt[surgeon, d, l] for surgeon in list_surgeon) + pulp.lpSum(surgery.get_priority() * ts[surgery, l] for surgery in waiting_list)
                operating_room_scheduling += pulp.lpSum(lam[t] for t in range(1, iterations + 1)) == 1, 'c0'
                for r in list_room:
                    list_surgery_in_r = dict_surgery_room[r]
                    for surgery1 in list_surgery_in_r:
                        for surgery2 in list_surgery_in_r:
                            if surgery1 != surgery2:
                                operating_room_scheduling += y[surgery1, surgery2, r, d, l] + y[surgery2, surgery1, r, d, l] == 1, 'c1_' + str(surgery1.get_surgery_id()) + '_' + str(surgery2.get_surgery_id()) + '_' + str(r)

                for surgeon in planned_surgeon:
                    list_surgery_by_k = dict_surgery_surgeon[surgeon]
                    for surgery1 in list_surgery_by_k:
                        for surgery2 in list_surgery_by_k:
                            if surgery1 != surgery2:
                                operating_room_scheduling += z[surgery1, surgery2, surgeon, d, l] + z[surgery2, surgery1, surgeon, d, l] == 1, 'c2_' + str(surgery1.get_surgery_id()) + '_' + str(surgery2.get_surgery_id()) + '_' + str(surgeon.get_surgeon_id())

                for surgery in planned_surgery:
                    operating_room_scheduling += pulp.lpSum(omega1[surgery, t] * lam[t] for t in range(1, iterations + 1)) - ts[surgery, l] <= 0, 'c3_' + str(surgery.get_surgery_id())

                operating_room_scheduling += pulp.lpSum(np.log(norm.cdf((np.log(omega1[surgery, t]) - mu1[surgery]) / np.sqrt(sigma1[surgery]))) * lam[t] for t in range(1, iterations + 1) for surgery in planned_surgery) >= np.log(alpha), 'c3_prime'

                for r in list_room:
                    list_surgery_in_r = dict_surgery_room[r]
                    for surgery1 in list_surgery_in_r:
                        for surgery2 in list_surgery_in_r:
                            if surgery1 != surgery2:
                                operating_room_scheduling += pulp.lpSum(omega2[surgery1, surgery2, t] * lam[t] for t in range(1, iterations + 1)) <= ts[surgery2, l] - ts[surgery1, l] + M * (1 - y[surgery1, surgery2, r, d, l]), 'c4_' + str(surgery1.get_surgery_id()) + '_' + str(surgery2.get_surgery_id()) + '_' + str(r)

                operating_room_scheduling += pulp.lpSum(np.log(norm.cdf((np.log(omega2[surgery1, surgery2, t]) - mu2[surgery1, surgery2]) / np.sqrt(sigma2[surgery1, surgery2]))) * lam[t] for t in range(1, iterations + 1) for r in list_room for surgery1 in dict_surgery_room[r] for surgery2 in dict_surgery_room[r] if surgery1 != surgery2) >= np.log(alpha), 'c4_prime'

                for surgeon in planned_surgeon:
                    list_surgery_by_k = dict_surgery_surgeon[surgeon]
                    for surgery1 in list_surgery_by_k:
                        for surgery2 in list_surgery_by_k:
                            if surgery1 != surgery2:
                                operating_room_scheduling += -ts[surgery2, l] + ts[surgery1, l] + PT - M * (1 - z[surgery1, surgery2, surgeon, d, l]) + pulp.lpSum(omega3[surgery1, surgery2, t] * lam[t] for t in range(1, iterations + 1)) <= 0, 'c5_' + str(surgery1.get_surgery_id()) + '_' + str(surgery2.get_surgery_id()) + '_' + str(surgeon.get_surgeon_id())

                operating_room_scheduling += pulp.lpSum(np.log(norm.cdf((np.log(omega3[surgery1, surgery2, t]) - mu3[surgery1, surgery2]) / np.sqrt(sigma3[surgery1, surgery2]))) * lam[t] for t in range(1, iterations + 1) for surgeon in planned_surgeon for surgery1 in dict_surgery_surgeon[surgeon] for surgery2 in dict_surgery_surgeon[surgeon] if surgery1 != surgery2) >= np.log(alpha), 'c5_prime'

                for surgeon in planned_surgeon:
                    for surgery in dict_surgery_surgeon[surgeon]:
                        operating_room_scheduling += tsS[surgeon, d, l] <= ts[surgery, l], 'c6_' + str(surgery.get_surgery_id()) + '_' + str(surgeon.get_surgeon_id())

                for r in list_room:
                    for surgery in dict_surgery_room[r]:
                        operating_room_scheduling += -msR[r, d, l] + ts[surgery, l] + pulp.lpSum(omega4[surgery, t] * lam[t] for t in range(1, iterations + 1)) <= 0, 'c7_' + str(surgery.get_surgery_id()) + '_' + str(r)

                operating_room_scheduling += pulp.lpSum(np.log(norm.cdf((np.log(omega4[surgery, t]) - mu4[surgery]) / np.sqrt(sigma4[surgery]))) * lam[t] for t in range(1, iterations + 1) for r in list_room for surgery in dict_surgery_room[r]) >= np.log(alpha), 'c7_prime'
                for surgeon in planned_surgeon:
                    for surgery in dict_surgery_surgeon[surgeon]:
                        operating_room_scheduling += -msS[surgeon, d, l] + ts[surgery, l] + pulp.lpSum(omega5[surgery, t] * lam[t] for t in range(1, iterations + 1)) <= 0, 'c8_' + str(surgery.get_surgery_id()) + '_' + str(surgeon.get_surgeon_id())

                operating_room_scheduling += pulp.lpSum(np.log(norm.cdf((np.log(omega5[surgery, t]) - mu5[surgery]) / np.sqrt(sigma5[surgery]))) * lam[t] for t in range(1, iterations + 1) for surgeon in planned_surgeon for surgery in dict_surgery_surgeon[surgeon]) >= np.log(alpha), 'c8_prime'
                for r in list_room:
                    operating_room_scheduling += ot[r, d, l] >= msR[r, d, l] - T, 'c9_' + str(r)

                """
                for r in list_room:
                    operating_room_scheduling += ot[r, d, l] <= O_max, 'c10_' + str(r)
                """
                for surgeon in planned_surgeon:
                    operating_room_scheduling += -pulp.lpSum(omega6[surgeon, t] * lam[t] for t in range(1, iterations + 1)) + msS[surgeon, d, l] - tsS[surgeon, d, l] - wt[surgeon, d, l] <= 0, 'c11_' + str(surgeon.get_surgeon_id())
                operating_room_scheduling += pulp.lpSum(np.log(1 - norm.cdf((np.log(omega6[surgeon, t]) - mu6[surgeon]) / np.sqrt(sigma6[surgeon]))) * lam[t] for t in range(1, iterations + 1) for surgeon in planned_surgeon) >= np.log(alpha), 'c11_prime'
                result_status2 = operating_room_scheduling.solve(solver)
                if j == 0 and pulp.LpStatus[result_status2] != "Infeasible":
                    pi_sigma = operating_room_scheduling.constraints['c0'].pi
                    pi_3 = {}
                    for surgery in planned_surgery:
                        pi_3[surgery, d] = operating_room_scheduling.constraints['c3_' + str(surgery.get_surgery_id())].pi

                    pi_alpha_3 = operating_room_scheduling.constraints['c3_prime'].pi
                    print('pi_alpha_3={:}'.format(pi_alpha_3))
                    pi_4 = {}
                    for r in list_room:
                        list_surgery_in_r = dict_surgery_room[r]
                        for surgery1 in list_surgery_in_r:
                            for surgery2 in list_surgery_in_r:
                                if surgery1 != surgery2:
                                    pi_4[surgery1, surgery2, r, d] = operating_room_scheduling.constraints['c4_' + str(surgery1.get_surgery_id()) + '_' + str(surgery2.get_surgery_id()) + '_' + str(r)].pi

                    pi_alpha_4 = operating_room_scheduling.constraints['c4_prime'].pi
                    pi_5 = {}
                    for surgeon in planned_surgeon:
                        list_surgery_by_k = dict_surgery_surgeon[surgeon]
                        for surgery1 in list_surgery_by_k:
                            for surgery2 in list_surgery_by_k:
                                if surgery1 != surgery2:
                                    pi_5[surgery1, surgery2, surgeon, d] = operating_room_scheduling.constraints['c5_' + str(surgery1.get_surgery_id()) + '_' + str(surgery2.get_surgery_id()) + '_' + str(surgeon.get_surgeon_id())].pi

                    pi_alpha_5 = operating_room_scheduling.constraints['c5_prime'].pi
                    pi_7 = {}
                    for r in list_room:
                        for surgery in dict_surgery_room[r]:
                            pi_7[surgery, r, d] = operating_room_scheduling.constraints['c7_' + str(surgery.get_surgery_id()) + '_' + str(r)].pi

                    pi_alpha_7 = operating_room_scheduling.constraints['c7_prime'].pi
                    pi_8 = {}
                    for surgeon in planned_surgeon:
                        for surgery in dict_surgery_surgeon[surgeon]:
                            pi_8[surgery, surgeon, d] = operating_room_scheduling.constraints['c8_' + str(surgery.get_surgery_id()) + '_' + str(surgeon.get_surgeon_id())].pi

                    pi_alpha_8 = operating_room_scheduling.constraints['c8_prime'].pi
                    pi_11 = {}
                    for surgeon in planned_surgeon:
                        pi_11[surgeon, d] = operating_room_scheduling.constraints['c11_' + str(surgeon.get_surgeon_id())].pi

                    pi_alpha_11 = operating_room_scheduling.constraints['c11_prime'].pi
                print('ORS=' + pulp.LpStatus[result_status2])

            if pulp.LpStatus[result_status2] == "Infeasible":
                print(pulp.LpStatus[result_status2])
                break

            else:

                def func(x):
                    print(x)
                    start = 0
                    end = len(omega1)
                    omega1_var = {}
                    for i, var in enumerate(x[start:end]):
                        surgery = planned_surgery[i]
                        omega1_var[surgery] = var
                    start = end
                    end = end + len(omega2)
                    omega2_var = {}
                    for i, var in enumerate(x[start:end]):
                        surgery1, surgery2 = omega2_index_list[i]
                        omega2_var[surgery1, surgery2] = var

                    start = end
                    end = end + len(omega3)
                    omega3_var = {}
                    for i, var in enumerate(x[start:end]):
                        surgery1, surgery2 = omega3_index_list[i]
                        omega3_var[surgery1, surgery2] = var
                    start = end
                    end = end + len(omega4)
                    omega4_var = {}
                    for i, var in enumerate(x[start:end]):
                        surgery = omega4_index_list[i]
                        omega4_var[surgery] = var
                    start = end
                    end = end + len(omega5)
                    omega5_var = {}
                    for i, var in enumerate(x[start:end]):
                        surgery = omega5_index_list[i]
                        omega5_var[surgery] = var
                    start = end
                    end = end + len(omega6)
                    omega6_var = {}
                    for i, var in enumerate(x[start:end]):
                        surgeon = omega6_index_list[i]
                        omega6_var[surgeon] = var
                    expr_list = []
                    for surgery in planned_surgery:
                        expr = -pi_3[surgery, d] - pi_alpha_3 * ((log_normal(omega1_var[surgery], mu1[surgery], sigma1[surgery])) / norm.cdf((np.log(omega1_var[surgery]) - mu1[surgery]) / np.sqrt(sigma1[surgery])))
                        expr_list.append(expr)
                    for r in list_room:
                        list_surgery_in_r = dict_surgery_room[r]
                        for surgery1 in list_surgery_in_r:
                            for surgery2 in list_surgery_in_r:
                                if surgery1 != surgery2:
                                    expr = -pi_4[surgery1, surgery2, r, d] - pi_alpha_4 * (log_normal(omega2_var[surgery1, surgery2], mu2[surgery1, surgery2], sigma2[surgery1, surgery2]) / norm.cdf((np.log(omega2_var[surgery1, surgery2]) - mu2[surgery1, surgery2]) / np.sqrt(sigma2[surgery1, surgery2])))
                                    expr_list.append(expr)
                    for surgeon in planned_surgeon:
                        list_surgery_by_k = dict_surgery_surgeon[surgeon]
                        for surgery1 in list_surgery_by_k:
                            for surgery2 in list_surgery_by_k:
                                if surgery1 != surgery2:
                                    expr = -pi_5[surgery1, surgery2, surgeon, d] - pi_alpha_5 * (log_normal(omega3_var[surgery1, surgery2], mu3[surgery1, surgery2], sigma3[surgery1, surgery2]) / norm.cdf((np.log(omega3_var[surgery1, surgery2]) - mu3[surgery1, surgery2]) / np.sqrt(sigma3[surgery1, surgery2])))
                                    expr_list.append(expr)
                    for r in list_room:
                        for surgery in dict_surgery_room[r]:
                            expr = -pi_7[surgery, r, d] - pi_alpha_7 * (log_normal(omega4_var[surgery], mu4[surgery], sigma4[surgery]) / norm.cdf((np.log(omega4_var[surgery]) - mu4[surgery]) / np.sqrt(sigma4[surgery])))
                            expr_list.append(expr)
                    for surgeon in planned_surgeon:
                        for surgery in dict_surgery_surgeon[surgeon]:
                            expr = pi_8[surgery, surgeon, d] + pi_alpha_8 * (log_normal(omega5_var[surgery], mu5[surgery], sigma5[surgery]) / norm.cdf((np.log(omega5_var[surgery]) - mu5[surgery]) / np.sqrt(sigma5[surgery])))
                            expr_list.append(expr)
                    for surgeon in planned_surgeon:
                        expr = -pi_11[surgeon, d] - pi_alpha_11 * (log_normal(omega6_var[surgeon], mu6[surgeon], sigma6[surgeon]) / (1 - norm.cdf((np.log(omega6_var[surgeon]) - mu6[surgeon]) / np.sqrt(sigma6[surgeon]))))
                        expr_list.append(expr)
                    return expr_list
                omega1_ini = {}
                if sum(pi_3.values()) > 0:
                    for surgery in planned_surgery:
                        mu = mu1[surgery]
                        sigma = sigma1[surgery]
                        omega1_ini[surgery] = np.exp(norm.ppf(alpha ** (pi_3[surgery, d] / (sum(pi_3.values())))) * np.sqrt(sigma) + mu)
                else:
                    for surgery in planned_surgery:
                        omega1_ini[surgery] = omega1[surgery, iterations]
                if sum(pi_4.values()) > 0:
                    omega2_ini = {}
                    for r in list_room:
                        surgery_in_r = dict_surgery_room[r]
                        for surgery1 in surgery_in_r:
                            for surgery2 in surgery_in_r:
                                if surgery1 != surgery2:
                                    mu = mu2[surgery1, surgery2]
                                    sigma = sigma2[surgery1, surgery2]
                                    omega2_ini[surgery1, surgery2] = np.exp(norm.ppf(alpha ** (pi_4[surgery1, surgery2, r, d] / sum(pi_4.values()))) * np.sqrt(sigma) + mu)
                else:
                    omega2_ini = omega2
                if sum(pi_5.values()) > 0:
                    omega3_ini = {}
                    for surgeon in planned_surgeon:
                        surgery_by_k = dict_surgery_surgeon[surgeon]
                        for surgery1 in surgery_by_k:
                            for surgery2 in surgery_by_k:
                                if surgery1 != surgery2:
                                    mu = mu3[surgery1, surgery2]
                                    sigma = sigma3[surgery1, surgery2]
                                    omega3_ini[surgery1, surgery2] = np.exp(norm.ppf(alpha ** (pi_5[surgery1, surgery2, surgeon, d] / sum(pi_5.values()))) * np.sqrt(sigma) + mu)

                else:
                    omega3_ini = omega3
                if sum(pi_7.values()) > 0:
                    omega4_ini = {}
                    for r in list_room:
                        surgery_in_r = dict_surgery_room[r]
                        for surgery in surgery_in_r:
                            mu = mu4[surgery]
                            sigma = sigma4[surgery]
                            omega4_ini[surgery] = np.exp(norm.ppf(alpha ** (pi_7[surgery, r, d] / sum(pi_7.values()))) * np.sqrt(sigma) + mu)

                else:
                    omega4_ini = omega4
                if sum(pi_8.values()) > 0:
                    omega5_ini = {}
                    for surgeon in planned_surgeon:
                        surgery_by_k = dict_surgery_surgeon[surgeon]
                        for surgery in surgery_by_k:
                            mu = mu5[surgery]
                            sigma = sigma5[surgery]
                            omega5_ini[surgery] = np.exp(norm.ppf(alpha ** (pi_8[surgery, surgeon, d] / sum(pi_8.values()))) * np.sqrt(sigma) + mu)
                else:
                    omega5_ini = omega5
                if sum(pi_11.values()) > 0:
                    omega6_ini = {}
                    for surgeon in planned_surgeon:
                        mu = mu6[surgeon]
                        sigma = sigma6[surgeon]
                        omega6[surgeon, 1] = np.exp(norm.ppf(1 - alpha ** (pi_11[surgeon, d] / sum(pi_11.values()))) * np.sqrt(sigma) + mu)

                else:
                    omega6_ini = omega6

                initial_value = list(omega1_ini.values()) + list(omega2_ini.values()) + list(omega3_ini.values()) + list(omega4_ini.values()) + list(omega5_ini.values()) + list(omega6_ini.values())
                # result = optimize.broyden2(func, initial_value)
                # print(result)
                # x = result.x
                s = planned_surgery[0]
                x = newton_method(omega1_ini[s], mu1[s], sigma1[s], pi_3[s, d], pi_alpha_3)
                print(x)
                start = 0
                end = len(omega1)
                start = 0
                end = len(omega1)
                for i, var in enumerate(x[start:end]):
                    surgery = planned_surgery[i]
                    omega1[surgery, iterations + 1] = var
                start = end
                end = end + len(omega2)
                for i, var in enumerate(x[start:end]):
                    surgery1, surgery2 = omega2_index_list[i]
                    omega2[surgery1, surgery2, iterations + 1] = var

                start = end
                end = end + len(omega3)
                for i, var in enumerate(x[start:end]):
                    surgery1, surgery2 = omega3_index_list[i]
                    omega3[surgery1, surgery2, iterations + 1] = var
                start = end
                end = end + len(omega4)
                for i, var in enumerate(x[start:end]):
                    surgery = omega4_index_list[i]
                    omega4[surgery, iterations + 1] = var
                start = end
                end = end + len(omega5)
                for i, var in enumerate(x[start:end]):
                    surgery = omega5_index_list[i]
                    omega5[surgery, iterations + 1] = var
                start = end
                end = end + len(omega6)
                for i, var in enumerate(x[start:end]):
                    surgeon = omega6_index_list[i]
                    omega6[surgeon, iterations + 1] = var

                obj_value = sum(pi_3[surgery, d] * omega1[surgery1.get_surgery_id(), iterations + 1] for surgery in planned_surgery)
                obj_value += sum(pi_4[surgery1, surgery2, r, d] * omega2[surgery1, surgery2, iterations + 1] for r in list_room for surgery1 in dict_surgery_room[r] for surgery2 in dict_surgery_room[r] if surgery1 != surgery2)
                obj_value += sum(pi_5[surgery1, surgery2, surgeon.get_surgeon_id(), d] * omega3[surgery1, surgery2, iterations + 1] for surgeon in planned_surgeon for surgery1 in dict_surgery_surgeon[surgeon] for surgery2 in dict_surgery_surgeon[surgeon] if surgery1 != surgery2)
                obj_value += sum(pi_7[surgery, r, d] * omega4[surgery, iterations + 1] for r in list_room for surgery in dict_surgery_room[r])
                obj_value += sum(pi_8[surgery, surgeon.get_surgeon_id(), d] * omega5[surgery, iterations + 1] for surgeon in planned_surgeon for surgery in dict_surgery_surgeon[surgeon])
                obj_value += sum(pi_11[surgeon, d] * omega6[surgeon, iterations + 1] for surgeon in planned_surgeon)
                if obj_value > stopping_criteria:
                    stopping_criteria = obj_value
                else:
                    feasibility_criteria2 = True

                """
                omega1_var = {}
                omega2_var = {}
                omega3_var = {}
                omega4_var = {}
                omega5_var = {}
                omega6_var = {}
                for s in planned_surgery:
                    omega1_var[s.get_surgery_id()] = sympy.Symbol('omega1({:})'.format(s.get_surgery_id()))
                for r in list_room:
                    list_surgery_in_r = dict_surgery_room[r]
                    for surgery1 in list_surgery_in_r:
                        for surgery2 in list_surgery_in_r:
                            if surgery1 != surgery2:
                                omega2_var[surgery1, surgery2] = sympy.Symbol('omega2({:},{:})'.format(surgery1.get_surgery_id(), surgery2.get_surgery_id()))

                for surgeon in planned_surgeon:
                    list_surgery_by_k = dict_surgery_surgeon[surgeon]
                    for surgery1 in list_surgery_by_k:
                        omega3_var[surgery1.get_surgery_id()] = sympy.Symbol('omega3({:})'.format(surgery1.get_surgery_id()))
                for r in list_room:
                    for surgery in dict_surgery_room[r]:
                        omega4_var[surgery] = sympy.Symbol('omega4({:})'.format(surgery.get_surgery_id()))

                for surgeon in planned_surgeon:
                    for surgery in dict_surgery_surgeon[surgeon]:
                        omega5_var[surgery] = sympy.Symbol('omega5({:})'.format(surgery.get_surgery_id()))

                for surgeon in planned_surgeon:
                    omega6_var[surgeon] = sympy.Symbol('omega6({:})'.format(surgeon.get_surgeon_id()))

                pi_alpha_1 = sympy.Symbol('pi_alpha_1')
                pi_alpha_2 = sympy.Symbol('pi_alpha_2')
                pi_alpha_3 = sympy.Symbol('pi_alpha_3')
                pi_alpha_4 = sympy.Symbol('pi_alpha_4')
                pi_alpha_5 = sympy.Symbol('pi_alpha_5')
                pi_alpha_6 = sympy.Symbol('pi_alpha_6')
                expr_list = []
                for surgery in planned_surgery:
                    mu = distribution_dict[surgery.get_group(), 'preparation_mu']
                    sigma = distribution_dict[surgery.get_group(), 'preparation_sigma']
                    expr = pi_3[surgery1.get_surgery_id(), d] + pi_alpha_1 * (1 / (np.sqrt(2 * np.pi * sigma) * omega1_var[surgery]) * sympy.exp(-(sympy.log(omega1_var[surgery]) - mu) ** 2 / (2 * sigma)) / aprox_norm_cdf((sympy.log(omega1_var[surgery]) - mu) / np.sqrt(sigma)))
                    expr_list.append(expr)
                for r in list_room:
                    list_surgery_in_r = dict_surgery_room[r]
                    for surgery1 in list_surgery_in_r:
                        for surgery2 in list_surgery_in_r:
                            if surgery1 != surgery2:
                                mean = distribution_dict[surgery1.get_group(), 'surgery_mean'] + distribution_dict[surgery1.get_group(), 'cleaning_mean'] + distribution_dict[surgery2.get_group(), 'preparation_mean']
                                variance = distribution_dict[surgery1.get_group(), 'surgery_variance'] + distribution_dict[surgery1.get_group(), 'cleaning_variance'] + distribution_dict[surgery2.get_group(), 'preparation_variance']
                                sigma = np.log(variance / (mean**2) + 1)
                                mu = np.log(mean) - sigma / 2
                                expr = pi_4[surgery1, surgery2, r, d] + pi_alpha_2 * (1 / (np.sqrt(2 * np.pi * sigma) * omega2_var[surgery1, surgery2]) * sympy.exp(-(sympy.log(omega2_var[surgery1, surgery2]) - mu) ** 2 / (2 * sigma)) / (aprox_norm_cdf((sympy.log(omega2_var[surgery1, surgery2]) - mu) / np.sqrt(sigma))))
                                expr_list.append(expr)
                for surgeon in planned_surgeon:
                    list_surgery_by_k = dict_surgery_surgeon[surgeon]
                    for surgery1 in list_surgery_by_k:
                        for surgery2 in list_surgery_by_k:
                            if surgery1 != surgery2:
                                mu = distribution_dict[surgery1.get_group(), 'surgery_mu']
                                sigma = distribution_dict[surgery1.get_group(), 'surgery_sigma']
                                expr = pi_5[surgery1, surgery2, surgeon.get_surgeon_id(), d] + pi_alpha_3 * (1 / (np.sqrt(2 * np.pi * sigma) * omega3_var[surgery1.get_surgery_id()]) * sympy.exp(-(sympy.log(omega3_var[surgery1.get_surgery_id()]) - mu) ** 2 / (2 * sigma)) / (aprox_norm_cdf((sympy.log(omega3_var[surgery1.get_surgery_id()]) - mu) / np.sqrt(sigma))))
                                expr_list.append(expr)
                for r in list_room:
                    for surgery in dict_surgery_room[r]:
                        mean = distribution_dict[surgery.get_group(), 'surgery_mean'] + distribution_dict[surgery.get_group(), 'cleaning_mean']
                        variance = distribution_dict[surgery.get_group(), 'surgery_variance'] + distribution_dict[surgery.get_group(), 'cleaning_variance']
                        sigma = np.log(variance / (mean**2) + 1)
                        mu = np.log(mean) - sigma / 2
                        expr = pi_7[surgery, r, d] + pi_alpha_4 * (1 / (np.sqrt(2 * np.pi * sigma) * omega4_var[surgery]) * sympy.exp(-(sympy.log(omega4_var[surgery]) - mu) ** 2 / (2 * sigma)) / (aprox_norm_cdf((sympy.log(omega4_var[surgery]) - mu) / np.sqrt(sigma))))
                        expr_list.append(expr)
                for surgeon in planned_surgeon:
                    for surgery in dict_surgery_surgeon[surgeon]:
                        mu = distribution_dict[surgery.get_group(), 'surgery_mu']
                        sigma = distribution_dict[surgery.get_group(), 'surgery_sigma']
                        expr = pi_8[surgery, surgeon.get_surgeon_id(), d] + pi_alpha_5 * (1 / (np.sqrt(2 * np.pi * sigma) * omega5_var[surgery]) * sympy.exp(-(sympy.log(omega5_var[surgery]) - mu) ** 2 / (2 * sigma)) / (aprox_norm_cdf((sympy.log(omega5_var[surgery]) - mu) / np.sqrt(sigma))))
                        expr_list.append(expr)

                for surgeon in planned_surgeon:
                    mean = sum(distribution_dict[surgery.get_group(), 'surgery_mean'] for surgery in dict_surgery_surgeon[surgeon])
                    variance = sum(distribution_dict[surgery.get_group(), 'surgery_variance'] for surgery in dict_surgery_surgeon[surgeon])
                    sigma = np.log(variance / (mean**2) + 1)
                    mu = np.log(mean) - sigma / 2
                    expr = pi_11[surgeon, d] + pi_alpha_6 * (1 / (np.sqrt(2 * np.pi * sigma) * omega6_var[surgeon]) * sympy.exp(-(sympy.log(omega6_var[surgeon]) - mu) ** 2 / (2 * sigma)) / (aprox_norm_cdf((sympy.log(omega6_var[surgeon]) - mu) / np.sqrt(sigma))))
                    expr_list.append(expr)

                expr1 = sum(sympy.log(aprox_norm_cdf((sympy.log(omega1_var[surgery]) - mu1[surgery1.get_surgery_id()]) / np.sqrt(sigma1[surgery1.get_surgery_id()]))) for surgery in planned_surgery) - sympy.log(alpha)
                expr2 = sum(sympy.log(aprox_norm_cdf((sympy.log(omega2_var[surgery1, surgery2]) - mu2[surgery1, surgery2]) / np.sqrt(sigma2[surgery1, surgery2]))) for r in list_room for surgery1 in dict_surgery_room[r] for surgery2 in dict_surgery_room[r] if surgery1 != surgery2) - np.log(alpha)
                expr3 = sum(sympy.log(aprox_norm_cdf((sympy.log(omega3_var[surgery1.get_surgery_id()]) - mu3[surgery1.get_surgery_id()]) / np.sqrt(sigma3[surgery1.get_surgery_id()]))) for surgeon in planned_surgeon for surgery1 in dict_surgery_surgeon[surgeon] for surgery2 in dict_surgery_surgeon[surgeon] if surgery1 != surgery2) - np.log(alpha)
                expr4 = sum(sympy.log(aprox_norm_cdf((sympy.log(omega4_var[surgery]) - mu4[surgery]) / np.sqrt(sigma4[surgery]))) for r in list_room for surgery in dict_surgery_room[r]) - np.log(alpha)
                expr5 = sum(sympy.log(aprox_norm_cdf((sympy.log(omega5_var[surgery]) - mu5[surgery]) / np.sqrt(sigma5[surgery]))) for surgeon in planned_surgeon for surgery in dict_surgery_surgeon[surgeon]) - np.log(alpha)
                expr6 = sum(sympy.log(aprox_norm_cdf((sympy.log(omega6_var[surgeon]) - mu6[surgeon]) / np.sqrt(sigma6[surgeon]))) for surgeon in planned_surgeon) - np.log(alpha)
                expr_list.append(expr1)
                expr_list.append(expr2)
                expr_list.append(expr3)
                expr_list.append(expr4)
                expr_list.append(expr5)
                expr_list.append(expr6)
                var_list = list(omega1_var.values()) + list(omega2_var.values()) + list(omega3_var.values()) + list(omega4_var.values()) + list(omega5_var.values()) + list(omega6_var.values()) + [pi_alpha_1, pi_alpha_2, pi_alpha_3, pi_alpha_4, pi_alpha_5, pi_alpha_6]
                print(var_list)
                sol = sympy.solve(expr_list)
                print(sol)
                """
                # 補助問題の求解
                # 最適性の判定
                # 最適->解を記録
                # 最適でない->補助問題の解を加えて解き直し。

    """
    print("式")
    print("--------")
    print(operating_room_scheduling)
    print("--------")
    print("")

    print("計算結果")
    print("********")

    print("最適性 = {:}, 目的関数値 = {:}"
          .format(pulp.LpStatus[result_status], pulp.value(operating_room_scheduling.objective)))
    """

    scheduled_list = []
    scheduled_surgeon_list = []
    for surgery in waiting_list:
        for r in list_room:
            if round(x[surgery, r, d, l].value()) == 1:
                scheduled_list.append(surgery)
        for surgeon in list_surgeon:
            if round(q[surgery, surgeon.get_surgeon_id(), d, l].value()) == 1:
                scheduled_surgeon_list.append(surgeon)
end = time.time()

rest_surgery = list(set(waiting_list + list_surgery_copied))
for surgery in rest_surgery:
    print(surgery.get_surgery_id(), surgery.get_group())
print("残り手術数={:}".format(len(rest_surgery)))
print("計算時間 = {:}".format(end - start))
print("手術数 = {:}".format(len(list_surgery)))
print("外科医の人数 = {:}".format(len(list_surgeon)))
def objective_function(lst_surgery, lst_surgeon, lst_room, lst_date):
    objective = pulp.lpSum(((d - surgery.get_release_date()) + max(d - surgery.get_due_date(), 0)) * x_val[surgery, r, d] * surgery.get_priority() for surgery in lst_surgery for r in lst_room for d in lst_date)
    objective += 10000 * pulp.lpSum(((surgery.get_due_date() - surgery.get_release_date()) + (len(lst_date) + 1 - surgery.get_due_date())) * surgery.get_priority() for surgery in lst_surgery) - pulp.lpSum(((surgery.get_due_date() - surgery.get_release_date()) + (len(lst_date) + 1 - surgery.get_due_date())) * surgery.get_priority() * x_val[surgery, r, d] for surgery in lst_surgery for r in lst_room for d in lst_date)
    objective += pulp.lpSum(n_val[surgeon, d] for surgeon in lst_surgeon for d in lst_date) + pulp.lpSum(ot_val[r, d] for r in lst_room for d in lst_date)
    objective += pulp.lpSum(wt_val[surgeon, d] for surgeon in lst_surgeon for d in lst_date) + pulp.lpSum(surgery.get_priority() * ts_val[surgery] for surgery in lst_surgery)
    return objective


objective_value = objective_function(list_surgery, list_surgeon, list_room, list_date)
print("目的関数値={:}".format(objective_value))
over_time = sum(ot_val[r, d] for r in list_room for d in list_date)
print("残業時間={:}".format(over_time))
book.save(url)
book.close()
book2.save(url2)
book2.close()
graph.create_ganttchart(url)
graph.create_surgeon_ganttchart(url2)
