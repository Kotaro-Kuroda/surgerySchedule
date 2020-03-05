import random
import os
import csv
import re
home = os.environ['HOME']
operations = []
with open(home + '/Documents/data-july/operations.csv') as f:
    reader = csv.reader(f)
    operations = [row for row in reader]
class Surgery:
    def __init__(self, surgery_id, preparation_time, surgery_time, cleaning_time, group, release_date, due_date, priority):
        self.surgery_id = surgery_id
        self.preparation_time = preparation_time
        self.surgery_time = surgery_time
        self.cleaning_time = cleaning_time
        self.group = group
        self.release_date = release_date
        self.due_date = due_date
        self.priority = priority

    def get_surgery_id(self):
        return self.surgery_id

    def get_preparation_time(self):
        return self.preparation_time

    def get_surgery_time(self):
        return self.surgery_time

    def get_cleaning_time(self):
        return self.cleaning_time

    def get_group(self):
        return self.group

    def get_release_date(self):
        return self.release_date

    def get_due_date(self):
        return self.due_date

    def get_priority(self):
        return self.priority

class Surgeon:
    def __init__(self, surgeon_id, group):
        self.surgeon_id = surgeon_id
        self.group = group

    def get_surgeon_id(self):
        return self.surgeon_id

    def get_group(self):
        return self.group

def getOccupancyDuration(str):
    hourexp = re.compile(r'\d+\s*hours')
    minexp = re.compile(r'\d+\s*min')
    numexp = re.compile(r'\d+')
    hour = int(numexp.search(hourexp.search(str).group()).group())
    minute = int(numexp.search(minexp.search(str).group()).group())
    return hour * 60 + minute

def getSurgeryTime(duration):
    return int(duration * 0.8)

def getPreparationTime(duration):
    return int(duration * 0.1)

def getCleaningTime(duration):
    return int(duration * 0.1)
def get_random_list(num, seed):
    random.seed(seed)
    list1 = []
    while len(list1) < num:
        ran = random.randint(1, len(operations) - 1)
        if ran not in list1:
            list1.append(ran)
    return list1

def create_model(num_surgery, num_date, seed):
    random.seed(seed)
    random_list = get_random_list(num_surgery, seed)
    list_surgery = []
    for i in random_list:
        surgery_id = operations[i][0]
        duration = getOccupancyDuration(operations[i][3])
        preparation_time = getPreparationTime(duration)
        surgery_time = getSurgeryTime(duration)
        cleaning_time = getCleaningTime(duration)
        group = operations[i][6]
        release_date = random.randint(1, num_date)
        due_date = min(release_date + random.randint(1, num_date), num_date)
        priority = 1 / (random.randint(2, 4) + (due_date - release_date))
        surgery = Surgery(surgery_id, preparation_time, surgery_time, cleaning_time, group, release_date, due_date, priority)
        list_surgery.append(surgery)

    list_surgeon_id = []
    for i in random_list:
        list_surgeon_id.append(operations[i][7])

    list_surgeon = []
    list_surgeon_id = list(set(list_surgeon_id))
    for surgeon_id in list_surgeon_id:
        for i in random_list:
            if surgeon_id == operations[i][7]:
                group = operations[i][6]
                surgeon = Surgeon(surgeon_id, group)
                list_surgeon.append(surgeon)
                break

    list_date = [i for i in range(1, num_date + 1)]
    return list_surgery, list_surgeon, list_date
