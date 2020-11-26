import csv
import surgery_class
import numpy as np
import datetime
import surgeon_class
class SurgeryInstance:
    def __init__(self, path, surgeon_info_path, num_surgery, num_date, seed, surgeries_path, distribution_path):
        self.path = path
        self.surgeon_info_path = surgeon_info_path
        self.num_surgery = num_surgery
        self.num_date = num_date
        self.seed = seed
        self.surgeries_path = surgeries_path
        self.distribution_path = distribution_path

    def get_csv(self):
        with open(self.path) as f:
            reader = csv.reader(f)
            csv_data = [row for row in reader]
        return csv_data

    def get_room_list(self, csv_data):
        list_room = sorted(list(set([int(csv_data[i][5]) for i in range(1, len(csv_data)) if csv_data[i][5] != ''])))
        list_room = [str(i) for i in list_room]
        return list_room

    def get_group_list(self, csv_data):
        list_group = list(set([csv_data[i][6] for i in range(1, len(csv_data))]))
        return list_group

    def get_room_surgery_dict(self):
        with open(self.surgeries_path) as f:
            reader = csv.reader(f)
            l = [row for row in reader]

        surgery_id = list(set([l[i][0] for i in range(1, len(l))]))
        dict1 = {}
        for s_id in surgery_id:
            dict1[s_id] = []

        for i in range(1, len(l)):
            s_id = l[i][0]
            dict1[s_id].append(l[i][1])
        return dict1

    def get_data(self, csv_data):
        np.random.seed(seed=self.seed)
        index_list = np.random.choice(range(1, len(csv_data)), self.num_surgery)
        data = [csv_data[i] for i in index_list]
        return data

    def get_delta_time(self, start, end):
        if start != '' and end != '':
            delta = datetime.datetime.strptime(end[:-3], '%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(start[:-3], '%Y-%m-%d %H:%M:%S')
            delta_min = int(delta.total_seconds() / 60)
            if delta_min > 0:
                return delta_min
            else:
                return 0
        else:
            return None

    def get_surgery_list(self, data):
        list_surgery = []
        for row in data:
            operation_id = row[0]
            group = row[6]
            preparation_time = self.get_delta_time(row[10], row[11])
            surgery_time = self.get_delta_time(row[11], row[12])
            if group == '心臓血管外科':
                if surgery_time > 0 and surgery_time <= 200:
                    group = group + '(low)'
                elif surgery_time > 200:
                    group = group + '(high)'

            distribution_dict = self.get_dict_distribution()
            preparation_mean = distribution_dict[group, 'preparation_mean']
            surgery_mean = distribution_dict[group, 'surgery_mean']
            cleaning_mean = distribution_dict[group, 'cleaning_mean']
            random_preparation = int(np.random.lognormal(distribution_dict[group, 'preparation_mu'], distribution_dict[group, 'preparation_sigma'], 1))
            random_surgery = int(np.random.lognormal(distribution_dict[group, 'surgery_mu'], distribution_dict[group, 'surgery_sigma'], 1))
            random_cleaning = int(np.random.lognormal(distribution_dict[group, 'cleaning_mu'], distribution_dict[group, 'cleaning_sigma'], 1))

            cleaning_time = self.get_delta_time(row[12], row[13])
            release_date = np.random.randint(1, self.num_date + 1)
            due_date = min(release_date + np.random.randint(1, self.num_date + 1), self.num_date + 1)
            priority = 1 / (np.random.randint(2, 5) + (due_date - release_date))
            surgery_type = row[8].split(",")
            surgery = surgery_class.Surgery(operation_id, preparation_time, surgery_time, cleaning_time, group, surgery_type, release_date, due_date, priority, preparation_mean, surgery_mean, cleaning_mean, random_preparation, random_surgery, random_cleaning)
            list_surgery.append(surgery)
        return list_surgery

    def get_surgeon_list(self, data):
        list_surgeon_id = []
        list_surgeon = []
        for row in data:
            surgeons_id = row[7].split(",")
            list_surgeon_id += surgeons_id
        with open(self.surgeon_info_path) as f:
            reader = csv.reader(f)
            surgeon_info = [row for row in reader]

        list_surgeon_id = set(list_surgeon_id)
        for i in range(1, len(surgeon_info)):
            row = surgeon_info[i]
            if row[0] in list_surgeon_id:
                dict_group = {}
                for j in range(1, len(row)):
                    dict_group[surgeon_info[0][j]] = row[j]
                surgeon = surgeon_class.Surgeon(row[0], dict_group)
                list_surgeon.append(surgeon)
        return list_surgeon

    def get_sets(self):
        csv_data = self.get_csv()
        data = self.get_data(csv_data)
        list_room = self.get_room_list(csv_data)
        list_surgery = self.get_surgery_list(data)
        list_surgeon = self.get_surgeon_list(data)
        list_date = [i + 1 for i in range(self.num_date)]
        list_group = self.get_group_list(csv_data)
        return list_room, list_surgery, list_surgeon, list_date, list_group

    def get_dict_distribution(self):
        with open(self.distribution_path) as f:
            reader = csv.reader(f)
            distribution = [row for row in reader]

        distribution_dict = {}
        for i in range(1, len(distribution)):
            row = distribution[i]
            group = row[0]
            preparation_mu = float(row[1])
            preparation_mean = float(row[2])
            preparation_sigma = float(row[3])
            preparation_variance = float(row[4])
            surgery_mu = float(row[5])
            surgery_mean = float(row[6])
            surgery_sigma = float(row[7])
            surgery_variance = float(row[8])
            cleaning_mu = float(row[9])
            cleaning_mean = float(row[10])
            cleaning_sigma = float(row[11])
            cleaning_variance = float(row[12])
            total_mu = float(row[13])
            total_mean = float(row[14])
            total_sigma = float(row[15])
            total_variance = float(row[16])
            distribution_dict[group, 'preparation_mu'] = preparation_mu
            distribution_dict[group, 'preparation_mean'] = preparation_mean
            distribution_dict[group, 'preparation_sigma'] = preparation_sigma
            distribution_dict[group, 'preparation_variance'] = preparation_variance
            distribution_dict[group, 'surgery_mu'] = surgery_mu
            distribution_dict[group, 'surgery_mean'] = surgery_mean
            distribution_dict[group, 'surgery_sigma'] = surgery_sigma
            distribution_dict[group, 'surgery_variance'] = surgery_variance
            distribution_dict[group, 'cleaning_mu'] = cleaning_mu
            distribution_dict[group, 'cleaning_mean'] = cleaning_mean
            distribution_dict[group, 'cleaning_sigma'] = cleaning_sigma
            distribution_dict[group, 'cleaning_variance'] = cleaning_variance
            distribution_dict[group, 'total_mu'] = total_mu
            distribution_dict[group, 'total_mean'] = total_mean
            distribution_dict[group, 'total_sigma'] = total_sigma
            distribution_dict[group, 'total_variance'] = total_variance
        return distribution_dict
