import copy
import csv
import os
import time
import graph
import distribution
import openpyxl
import pulp
import create_instance
import numpy as np
from scipy.stats import norm
import datetime
import random
import surgery_class
import surgeon_class
# 手術クラス


class SurgerySchedule:
    def __init__(self, path, file_path, start_date, durations, shift_length=480, pause_time=15):
        self.path = path
        self.file_path = file_path
        self.start_date = start_date
        self.durations = durations
        self.shift_length = shift_length
        self.pause_time = pause_time

    def get_csv_data(self):
        with open(self.path) as f:
            reader = csv.reader(f)
            csv_data = [row for row in reader]
        return csv_data

    def get_data(self):
        csv_data = self.get_csv_data()
        data = []
        list_date = []
        list_room = []
        list_group = []
        for i in range(1, len(csv_data)):
            room = csv_data[i][5]
            group = csv_data[i][6]
            if group != '' and group not in list_group:
                list_group.append(group)
            if room != '' and room not in list_room:
                list_room.append(room)
        date = datetime.datetime.strptime(self.start_date, "%Y-%m-%d")
        start_point = 0
        for i in range(1, len(csv_data)):
            if csv_data[i][10].startswith(self.start_date):
                start_point = i
                break
        for i in range(self.durations):
            next_date = date + datetime.timedelta(days=i)
            string_next_date = next_date.strftime('%Y-%m-%d')
            list_date.append(i + 1)
            for j in range(start_point, len(csv_data)):
                if csv_data[j][10].startswith(string_next_date):
                    data.append(csv_data[j])
                else:
                    start_point = j
                    break
        return data, list_date, list_room, list_group

    def get_delta_time(self, start, end):
        if start != '' and end != '':
            delta = datetime.datetime.strptime(end[:-3], '%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(start[:-3], '%Y-%m-%d %H:%M:%S')
            delta_min = int(delta.total_seconds() / 60)
        return delta_min

    def get_sets(self):
        data, list_date, list_room, list_group = self.get_data()
        print(data)
        list_surgeon = []
        list_surgery = []
        for row in data:
            operation_id = row[0]
            surgeon_id = row[7]
            group = row[6]
            preparation_time = self.get_delta_time(row[10], row[11])
            surgery_time = self.get_delta_time(row[11], row[12])
            cleaning_time = self.get_delta_time(row[12], row[13])
            release_date = random.randint(1, len(list_date))
            due_date = min(release_date + random.randint(1, len(list_date)), len(list_date))
            priority = 1 / (random.randint(2, 4) + (due_date - release_date))
            surgery_type = row[8].split(',')
            surgery = surgery_class.Surgery(operation_id, preparation_time, surgery_time, cleaning_time, group, surgery_type, release_date, due_date, priority)
            surgeon = surgeon_class.Surgeon(surgeon_id, group)
            list_surgery.append(surgery)
            list_surgeon.append(surgeon)
        return list_surgery, list_surgeon, list_date, list_room, list_group

    def operations_room_planning_and_scheduling(self):
        list_surgery, list_surgeon, list_date, list_room, list_group = self.get_sets()
        list_surgery_copied = copy.copy(list_surgery)
        waiting_list = []
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
        for d in list_date:
            feasiblity_criteria = False
            if len(list_surgery_copied) == 0:
                break
            for s in list_surgery_copied:
                if s.get_release_date() <= d:
                    if s not in waiting_list:
                        waiting_list.append(s)
                        list_surgery_copied.remove(s)
            loop = 0
            while not feasiblity_criteria:
                loop += 1
                for surgery in waiting_list:
                    for r in list_room:
                        x[surgery.get_surgery_id(), r, d, loop] = pulp.LpVariable("x({:},{:},{:},{:})".format(waiting_list.index(surgery), list_room.index(r), list_date.index(d), loop), cat='Binary')

                for surgery in waiting_list:
                    for surgeon in list_surgeon:
                        q[surgery.get_surgery_id(), surgeon.get_surgeon_id(), d, loop] = pulp.LpVariable("q({:},{:},{:},{:})".format(waiting_list.index(surgery), list_surgeon.index(surgeon), list_date.index(d), loop), cat='Binary')

                for surgery1 in waiting_list:
                    for surgery2 in waiting_list:
                        if surgery1 != surgery2:
                            for r in list_room:
                                y[surgery1.get_surgery_id(), surgery2.get_surgery_id(), r, d, loop] = pulp.LpVariable("y({:},{:},{:},{:},{:})".format(waiting_list.index(surgery1), waiting_list.index(surgery2), list_room.index(r), list_date.index(d), loop), cat='Binary')

                for surgery1 in waiting_list:
                    for surgery2 in waiting_list:
                        if surgery1 != surgery2:
                            for surgeon in list_surgeon:
                                z[surgery1.get_surgery_id(), surgery2.get_surgery_id(), surgeon.get_surgeon_id(), d, loop] = pulp.LpVariable("z({:},{:},{:},{:},{:})".format(waiting_list.index(surgery1), waiting_list.index(surgery2), list_surgeon.index(surgeon), list_date.index(d), loop), cat='Binary')

                for surgeon in list_surgeon:
                    n[surgeon.get_surgeon_id(), d, loop] = pulp.LpVariable("n({:},{:},{:})".format(list_surgeon.index(surgeon), list_date.index(d), loop), cat='Binary')

                for surgery in waiting_list:
                    ts[surgery.get_surgery_id(), loop] = pulp.LpVariable("ts({:},{:})".format(waiting_list.index(surgery), loop), lowBound=0, cat='Continuous')

                for r in list_room:
                    msR[r, d, loop] = pulp.LpVariable("msR({:},{:},{:})".format(list_room.index(r), list_date.index(d), loop), lowBound=0, cat='Continuous')

                for r in list_room:
                    ot[r, d, loop] = pulp.LpVariable("ot({:},{:},{:})".format(list_room.index(r), list_date.index(d), loop), lowBound=0, cat='Continuous')

                for surgeon in list_surgeon:
                    tsS[surgeon.get_surgeon_id(), d, loop] = pulp.LpVariable("tsS({:},{:},{:})".format(list_surgeon.index(surgeon), list_date.index(d), loop), lowBound=0, cat='Continuous')

                for surgeon in list_surgeon:
                    msS[surgeon.get_surgeon_id(), d, loop] = pulp.LpVariable("msS({:},{:},{:})".format(list_surgeon.index(surgeon), list_date.index(d), loop), lowBound=0, cat='Continuous')

                for surgeon in list_surgeon:
                    wt[surgeon.get_surgeon_id(), d, loop] = pulp.LpVariable("wt({:},{:},{:})".format(list_surgeon.index(surgeon), list_date.index(d), loop), lowBound=0, cat='Continuous')

                for surgeon in list_surgeon:
                    ost[surgeon.get_surgeon_id(), d, loop] = pulp.LpVariable("ost({:},{:},{:})".format(list_surgeon.index(surgeon), list_date.index(d), loop), lowBound=0, cat='Continuous')

                ns[d, loop] = pulp.LpVariable("ns({:},{:})".format(d, loop), lowBound=0, cat='Continuous')

                operating_room_planning = pulp.LpProblem("ORP" + str(loop), pulp.LpMinimize)
                objective = pulp.lpSum(((d - surgery.get_release_date()) + max(d - surgery.get_due_date(), 0)) * x[surgery.get_surgery_id(), r, d, loop] * surgery.get_priority() for surgery in waiting_list for r in list_room)
                objective += (pulp.lpSum(((surgery.get_due_date() - surgery.get_release_date()) + (len(list_date) + 1 - surgery.get_due_date())) * surgery.get_priority() for surgery in waiting_list) - pulp.lpSum(((surgery.get_due_date() - surgery.get_release_date()) + (len(list_date) + 1 - surgery.get_due_date())) * surgery.get_priority() * x[surgery.get_surgery_id(), r, d, loop] for surgery in waiting_list for r in list_room))
                objective += 1000 * (pulp.lpSum(surgery.get_priority() for surgery in waiting_list) - pulp.lpSum(surgery.get_priority() * x[surgery.get_surgery_id(), r, d, loop] for surgery in waiting_list for r in list_room))
                objective += pulp.lpSum(n[surgeon.get_surgeon_id(), d, loop] for surgeon in list_surgeon) + pulp.lpSum(ot[r, d, loop] for r in list_room) + pulp.lpSum(ost[surgeon.get_surgeon_id(), d, loop] for surgeon in list_surgeon) + 10000 * (ns[d, loop])
                operating_room_planning += objective
                for surgery in waiting_list:
                    operating_room_planning += pulp.lpSum(x[surgery.get_surgery_id(), r, d, loop] for r in list_room) <= 1
                for surgery in waiting_list:
                    operating_room_planning += pulp.lpSum(q[surgery.get_surgery_id(), surgeon.get_surgeon_id(), d, loop] for surgeon in list_surgeon) <= 1

                for surgery in waiting_list:
                    operating_room_planning += pulp.lpSum(x[surgery.get_surgery_id(), r, d, loop] for r in list_room) == pulp.lpSum(q[surgery.get_surgery_id(), surgeon.get_surgeon_id(), d, loop] for surgeon in list_surgeon)

                for surgeon in list_surgeon:
                    operating_room_planning += pulp.lpSum(q[surgery.get_surgery_id(), surgeon.get_surgeon_id(), d, loop] for surgery in waiting_list) <= len(waiting_list) * n[surgeon.get_surgeon_id(), d, loop]

                for surgeon in list_surgeon:
                    operating_room_planning += n[surgeon.get_surgeon_id(), d, loop] <= pulp.lpSum(q[surgery.get_surgery_id(), surgeon.get_surgeon_id(), d, loop] for surgery in waiting_list)

                for surgeon in list_surgeon:
                    operating_room_planning += pulp.lpSum(q[surgery.get_surgery_id(), surgeon.get_surgeon_id(), d, loop] for surgery in waiting_list) <= 3

                for r in list_room:
                    operating_room_planning += pulp.lpSum((surgery.get_preparation_time() + surgery.get_surgery_time() + surgery.get_cleaning_time()) * x[surgery.get_surgery_id(), r, d, loop] for surgery in waiting_list) - self.shift_length <= ot[r, d, loop]
        """
        for g in G:
            for r in dict_room_group[g]:
                surgery_group = list(set(waiting_list) & set(dict_surgery_group[g]))
                if len(surgery_group) > 0:
                    operating_room_planning += 0.95 * dict_room_group_prob[g, r] <= pulp.lpSum(x[surgery.get_surgery_id(), r, d, l] for surgery in surgery_group) / len(surgery_group)
                    operating_room_planning += 1.05 * dict_room_group_prob[g, r] >= pulp.lpSum(x[surgery.get_surgery_id(), r, d, l] for surgery in surgery_group) / len(surgery_group)
        """
        for g in G:
            for surgery in list(set(dict_surgery_group[g]) & set(waiting_list)):
                for r in list(set(R) - set(dict_room_group[g])):
                    operating_room_planning += x[surgery.get_surgery_id(), r, d, l] == 0

        for g in G:
            for surgery in list(set(dict_surgery_group[g]) & set(waiting_list)):
                for surgeon in list(set(list_surgeon) - set(dict_surgeon_group[g])):
                    operating_room_planning += q[surgery.get_surgery_id(), surgeon.get_surgeon_id(), d, l] == 0

        for r in R:
            operating_room_planning += ot[r, d, l] <= O_max

        for surgeon in list_surgeon:
            operating_room_planning += ost[surgeon.get_surgeon_id(), d, l] >= pulp.lpSum(TS[surgery.get_surgery_id()] * q[surgery.get_surgery_id(), surgeon.get_surgeon_id(), d, l] for surgery in waiting_list) - 1 / len(list_surgeon) * pulp.lpSum(TS[surgery.get_surgery_id()] for sugery in waiting_list)

        for surgeon in list_surgeon:
            operating_room_planning += -ost[surgeon.get_surgeon_id(), d, l] <= pulp.lpSum(TS[surgery.get_surgery_id()] * q[surgery.get_surgery_id(), surgeon.get_surgeon_id(), d, l] for surgery in waiting_list) - 1 / len(list_surgeon) * pulp.lpSum(TS[surgery.get_surgery_id()] for sugery in waiting_list)

        operating_room_planning += ns[d, l] >= pulp.lpSum(x[surgery.get_surgery_id(), r, d, l] for surgery in waiting_list for r in R) - int(len(list_surgery) / len(D))
        operating_room_planning += -ns[d, l] <= pulp.lpSum(x[surgery.get_surgery_id(), r, d, l] for surgery in waiting_list for r in R) - int(len(list_surgery) / len(D))
        if l > 1:
            for i in range(1, l):
                for surgery1 in waiting_list:
                    for surgery2 in waiting_list:
                        if surgery1 != surgery2:
                            operating_room_planning += pulp.lpSum(x[surgery1.get_surgery_id(), r, d, l] for r in R if x[surgery1.get_surgery_id(), r, d, i].value() == 1) + pulp.lpSum(x[surgery2.get_surgery_id(), r, d, l] for r in R if x[surgery2.get_surgery_id(), r, d, i].value() == 1) + pulp.lpSum(q[surgery1.get_surgery_id(), surgeon.get_surgeon_id(), d, l] for surgeon in list_surgeon if q[surgery1.get_surgery_id(), surgeon.get_surgeon_id(), d, i].value() == 1) + pulp.lpSum(q[surgery2.get_surgery_id(), surgeon.get_surgeon_id(), d, l] for surgeon in list_surgeon if q[surgery2.get_surgery_id(), surgeon.get_surgeon_id(), d, i].value() == 1) <= 3

        result_status = operating_room_planning.solve(solver)
        print(pulp.LpStatus[result_status])
        operating_room_scheduling = pulp.LpProblem("ORS", pulp.LpMinimize)
        operating_room_scheduling += pulp.lpSum(ot[r, d, l] for r in R) + pulp.lpSum(wt[surgeon.get_surgeon_id(), d, l] for surgeon in list_surgeon) + pulp.lpSum(U[surgery.get_surgery_id()] * ts[surgery.get_surgery_id(), l] for surgery in waiting_list)

        for surgery1 in waiting_list:
            for surgery2 in waiting_list:
                if surgery1 != surgery2:
                    for r in R:
                        if x[surgery1.get_surgery_id(), r, d, l].value() == 1 and x[surgery2.get_surgery_id(), r, d, l].value() == 1:
                            operating_room_scheduling += y[surgery1.get_surgery_id(), surgery2.get_surgery_id(), r, d, l] + y[surgery2.get_surgery_id(), surgery1.get_surgery_id(), r, d, l] == 1

        for surgery1 in waiting_list:
            for surgery2 in waiting_list:
                if surgery1 != surgery2:
                    for surgeon in list_surgeon:
                        if q[surgery1.get_surgery_id(), surgeon.get_surgeon_id(), d, l].value() == 1 and q[surgery2.get_surgery_id(), surgeon.get_surgeon_id(), d, l].value() == 1:
                            operating_room_scheduling += z[surgery1.get_surgery_id(), surgery2.get_surgery_id(), surgeon.get_surgeon_id(), d, l] + z[surgery2.get_surgery_id(), surgery1.get_surgery_id(), surgeon.get_surgeon_id(), d, l] == 1

        for surgery in waiting_list:
            operating_room_scheduling += ts[surgery.get_surgery_id(), l] >= TP[surgery.get_surgery_id()]

        for surgery1 in waiting_list:
            for surgery2 in waiting_list:
                if surgery1 != surgery2:
                    for r in R:
                        if x[surgery1.get_surgery_id(), r, d, l].value() == 1 and x[surgery2.get_surgery_id(), r, d, l].value() == 1:
                            operating_room_scheduling += ts[surgery2.get_surgery_id(), l] - TP[surgery2.get_surgery_id()] >= ts[surgery1.get_surgery_id(), l] + TS[surgery1.get_surgery_id()] + TC[surgery1.get_surgery_id()] - M * (1 - y[surgery1.get_surgery_id(), surgery2.get_surgery_id(), r, d, l])

        for surgery1 in waiting_list:
            for surgery2 in waiting_list:
                if surgery1 != surgery2:
                    for surgeon in list_surgeon:
                        if q[surgery1.get_surgery_id(), surgeon.get_surgeon_id(), d, l].value() == 1 and q[surgery2.get_surgery_id(), surgeon.get_surgeon_id(), d, l].value() == 1:
                            operating_room_scheduling += ts[surgery2.get_surgery_id(), l] >= ts[surgery1.get_surgery_id(), l] + TS[surgery1.get_surgery_id()] + PT - M * (1 - z[surgery1.get_surgery_id(), surgery2.get_surgery_id(), surgeon.get_surgeon_id(), d, l])

        for surgery in waiting_list:
            for surgeon in list_surgeon:
                if q[surgery.get_surgery_id(), surgeon.get_surgeon_id(), d, l].value() == 1:
                    operating_room_scheduling += tsS[surgeon.get_surgeon_id(), d, l] <= ts[surgery.get_surgery_id(), l]

        for surgery in waiting_list:
            for r in R:
                if x[surgery.get_surgery_id(), r, d, l].value() == 1:
                    operating_room_scheduling += msR[r, d, l] >= ts[surgery.get_surgery_id(), l] + TS[surgery.get_surgery_id()] + TC[surgery.get_surgery_id()]

        for surgery in waiting_list:
            for surgeon in list_surgeon:
                if q[surgery.get_surgery_id(), surgeon.get_surgeon_id(), d, l].value() == 1:
                    operating_room_scheduling += msS[surgeon.get_surgeon_id(), d, l] >= ts[surgery.get_surgery_id(), l] + TS[surgery.get_surgery_id()]

        for r in R:
            operating_room_scheduling += ot[r, d, l] >= msR[r, d, l] - T

        for r in R:
            operating_room_scheduling += ot[r, d, l] <= O_max

        for surgeon in list_surgeon:
            operating_room_scheduling += wt[surgeon.get_surgeon_id(), d, l] >= msS[surgeon.get_surgeon_id(), d, l] - tsS[surgeon.get_surgeon_id(), d, l] - pulp.lpSum(TS[surgery.get_surgery_id()] * q[surgery.get_surgery_id(), surgeon.get_surgeon_id(), d, l].value() for surgery in waiting_list)

        result_status2 = operating_room_scheduling.solve(solver)
        if pulp.LpStatus[result_status2] == "Infeasible":
            print(pulp.LpStatus[result_status2])
            continue
        else:
            feasiblity_criteria = True
            L.append(l)
            for surgery in waiting_list:
                for r in R:
                    if x[surgery.get_surgery_id(), r, d, l].value() == 1:
                        x_val[surgery.get_surgery_id(), r, d] = 1
                        ts_val[surgery.get_surgery_id()] = ts[surgery.get_surgery_id(), l].value()

            for r in R:
                ot_val[r, d] = ot[r, d, l].value()
            for surgeon in list_surgeon:
                wt_val[surgeon.get_surgeon_id(), d] = wt[surgeon.get_surgeon_id(), d, l].value()
                if n[surgeon.get_surgeon_id(), d, l].value() == 1:
                    n_val[surgeon.get_surgeon_id(), d] = 1
    def operation_room_planning(self, list_surgery, list_surgeon, list_date, list_room, list_group, loop):
        x = {}
home = os.environ['HOME']
path = home + '/list_dateocuments/data-july/operations_with_jisseki_remake.csv'
surgery_schedule = SurgerySchedule(path, '', "2019-06-03", 1)