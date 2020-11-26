import csv
import re
import os
import matplotlib as mpl
import numpy as np
mpl.rcParams['font.family'] = "IPAexGothic"

home = os.environ['HOME']
path = '/Documents/data-july/operations.csv'
l = []
with open(home + path) as f:
    reader = csv.reader(f)
    l = [row for row in reader]
group = []
for i in range(1, len(l)):
    if l[i][6] != '':
        group.append(l[i][6])

group = list(set(group))
def getOccupancyDuration(str):
    hourexp = re.compile(r'\d+\s*hours')
    minexp = re.compile(r'\d+\s*min')
    numexp = re.compile(r'\d+')
    hour = int(numexp.search(hourexp.search(str).group()).group())
    minute = int(numexp.search(minexp.search(str).group()).group())
    return hour * 60 + minute


def get_group_total_distribution(g):
    list_surgerytime = []
    for i in range(1, len(l)):
        if g == l[i][6]:
            if l[i][3] != "":
                delta_min = getOccupancyDuration(l[i][3])
                if delta_min > 0:
                    list_surgerytime.append(delta_min)
    mu = round(sum(i for i in list_surgerytime) / len(list_surgerytime), 2)
    sigma = round(sum((i - mu) ** 2 for i in list_surgerytime) / len(list_surgerytime), 2)
    return [mu, sigma]

def get_group_surgery_time_distribution(g):
    list_surgerytime = []
    for i in range(1, len(l)):
        if g == l[i][6]:
            if l[i][3] != "":
                delta_min = getOccupancyDuration(l[i][3])
                if delta_min > 0:
                    list_surgerytime.append(delta_min)
    mu = round(sum(np.log(i * 0.7) for i in list_surgerytime) / len(list_surgerytime), 2)
    sigma = round(sum((np.log(i * 0.7) - mu) ** 2 for i in list_surgerytime) / len(list_surgerytime), 2)
    u = np.exp(mu + sigma / 2)
    d = np.exp(2 * mu + sigma) * (np.exp(sigma) - 1)
    return [u, d]


def get_group_preparation_distribution(g):
    list_surgerytime = []
    for i in range(1, len(l)):
        if g == l[i][6]:
            if l[i][3] != "":
                delta_min = getOccupancyDuration(l[i][3])
                if delta_min > 0:
                    list_surgerytime.append(delta_min)
    mu = round(sum(np.log(i * 0.2) for i in list_surgerytime) / len(list_surgerytime), 2)
    sigma = round(sum((np.log(i * 0.2) - mu) ** 2 for i in list_surgerytime) / len(list_surgerytime), 2)
    u = np.exp(mu + sigma / 2)
    d = np.exp(2 * mu + sigma) * (np.exp(sigma) - 1)
    return [u, d]


def get_group_cleaning_distribution(g):
    list_surgerytime = []
    for i in range(1, len(l)):
        if g == l[i][6]:
            if l[i][3] != "":
                delta_min = getOccupancyDuration(l[i][3])
                if delta_min > 0:
                    list_surgerytime.append(delta_min)
    mu = round(sum(np.log(i * 0.1) for i in list_surgerytime) / len(list_surgerytime), 2)
    sigma = round(sum((np.log(i * 0.1) - mu) ** 2 for i in list_surgerytime) / len(list_surgerytime), 2)
    u = np.exp(mu + sigma / 2)
    d = np.exp(2 * mu + sigma) * (np.exp(sigma) - 1)
    return [u, d]
