import csv
import os
import matplotlib as mpl
import numpy as np
import datetime

mpl.rcParams['font.family'] = 'IPAexGothic'


class CsvDistribution:
    def __init__(self, path, file_path):
        self.path = path
        self.file_path = file_path

    def get_csv_data(self):
        with open(self.path) as f:
            reader = csv.reader(f)
            csv_data = [row for row in reader]
        return csv_data

    def get_delta_time(self, start, end):
        if start != '' and end != '':
            delta = datetime.datetime.strptime(end[:-3], '%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(start[:-3], '%Y-%m-%d %H:%M:%S')
            delta_min = int(delta.total_seconds() / 60)
            return delta_min
        else:
            return 0

    def get_group_list(self):
        csv_data = self.get_csv_data()
        G = []
        for i in range(1, len(csv_data)):
            g = csv_data[i][6]
            if g != '' and g not in G:
                G.append(g)
        return G

    def get_time_list(self, kind):
        G = self.get_group_list()
        csv_data = self.get_csv_data()
        dict_time_group = {}
        for g in G:
            dict_time_group[g] = []
        if kind == 'preparation':
            for i in range(1, len(csv_data)):
                g = csv_data[i][6]
                if g in G:
                    preparation_time = self.get_delta_time(csv_data[i][10], csv_data[i][11])
                    if preparation_time > 0:
                        dict_time_group[g].append(preparation_time)
        elif kind == 'surgery':
            for i in range(1, len(csv_data)):
                g = csv_data[i][6]
                if g in G:
                    surgery_time = self.get_delta_time(csv_data[i][11], csv_data[i][12])
                    if surgery_time > 0:
                        dict_time_group[g].append(surgery_time)
        elif kind == 'cleaning':
            for i in range(1, len(csv_data)):
                g = csv_data[i][6]
                if g in G:
                    cleaning_time = self.get_delta_time(csv_data[i][12], csv_data[i][13])
                    if cleaning_time > 0:
                        dict_time_group[g].append(cleaning_time)
        elif kind == 'total':
            for i in range(1, len(csv_data)):
                g = csv_data[i][6]
                if g in G:
                    total_time = self.get_delta_time(csv_data[i][10], csv_data[i][13])
                    if total_time > 0:
                        dict_time_group[g].append(total_time)
        else:
            return 'preparationかsurgeryかcleaningを指定してください'
        return dict_time_group

    def get_group_distribution(self, kind):
        dict_time_group = self.get_time_list(kind)
        dict_group_distribution = {}
        G = self.get_group_list()
        for g in G:
            list1 = dict_time_group[g]
            mu = round(sum(np.log(i) for i in list1) / len(list1), 2)
            sigma = round(sum((np.log(i) - mu) ** 2 for i in list1) / len(list1), 2)
            u = np.exp(mu + sigma / 2)
            d = np.exp(2 * mu + sigma) * (np.exp(sigma) - 1)
            dict_group_distribution[g, 'mu'] = mu
            dict_group_distribution[g, 'sigma'] = sigma
            dict_group_distribution[g, 'mean'] = u
            dict_group_distribution[g, 'variance'] = d
        return dict_group_distribution

    def make_csv_file(self):
        dict_preparation_distribution = self.get_group_distribution('preparation')
        dict_surgery_distribution = self.get_group_distribution('surgery')
        dict_cleaning_distribution = self.get_group_distribution('cleaning')
        dict_total_distribution = self.get_group_distribution('total')
        G = self.get_group_list()
        with open(self.file_path, 'w') as f:
            writer = csv.writer(f)
            title_row = ['group', 'preparation_mu', 'preparation_mean', 'preparation_sigma', 'preparation_variance', 'surgery_mu', 'surgery_mean', 'surgery_sigma', 'surgery_variance', 'cleaning_mu', 'cleaning_mean', 'cleaning_sigma', 'cleaning_variance', 'total_mu', 'total_mean', 'total_sigma', 'total_variance']
            writer.writerow(title_row)
            for g in G:
                preparation_mean = dict_preparation_distribution[g, 'mean']
                preparation_variance = dict_preparation_distribution[g, 'variance']
                surgery_mean = dict_surgery_distribution[g, 'mean']
                surgery_variance = dict_surgery_distribution[g, 'variance']
                cleaning_mean = dict_cleaning_distribution[g, 'mean']
                cleaning_variance = dict_cleaning_distribution[g, 'variance']
                preparation_mu = dict_preparation_distribution[g, 'mu']
                preparation_sigma = dict_preparation_distribution[g, 'sigma']
                surgery_mu = dict_surgery_distribution[g, 'mu']
                surgery_sigma = dict_surgery_distribution[g, 'sigma']
                cleaning_mu = dict_cleaning_distribution[g, 'mu']
                cleaning_sigma = dict_cleaning_distribution[g, 'sigma']
                total_mu = dict_total_distribution[g, 'mu']
                total_sigma = dict_total_distribution[g, 'sigma']
                total_mean = dict_total_distribution[g, 'mean']
                total_variance = dict_total_distribution[g, 'variance']
                row = [g, preparation_mu, preparation_mean, preparation_sigma, preparation_variance, surgery_mu, surgery_mean, surgery_sigma, surgery_variance, cleaning_mu, cleaning_mean, cleaning_sigma, cleaning_variance, total_mu, total_mean, total_sigma, total_variance]
                writer.writerow(row)


home = os.environ['HOME']
path = home + '/Documents/data-july/operations_with_jisseki_remake.csv'
file_path = home + '/Documents/surgerySchedule/distribution.csv'
csv_distribution = CsvDistribution(path, file_path)
csv_distribution.make_csv_file()
