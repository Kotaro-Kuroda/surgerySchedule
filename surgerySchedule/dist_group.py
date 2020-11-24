import numpy as np
import csv
import matplotlib.pyplot as plt
import os

home = os.environ['HOME']
file_path = home + '/Documents/surgerySchedule/distribution.csv'
with open(file_path) as f:
    reader = csv.reader(f)
    csv_data = [row for row in reader]
