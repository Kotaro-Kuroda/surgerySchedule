class Surgery:
    def __init__(self, surgery_id, preparation_time, surgery_time, cleaning_time, group, surgery_type, release_date, due_date, priority, preparation_mean, surgery_mean, cleaning_mean, random_preparation, random_surgery, random_cleaning):
        self.surgery_id = surgery_id
        self.preparation_time = preparation_time
        self.surgery_time = surgery_time
        self.cleaning_time = cleaning_time
        self.group = group
        self.surgery_type = surgery_type
        self.release_date = release_date
        self.due_date = due_date
        self.priority = priority
        self.preparation_mean = preparation_mean
        self.surgery_mean = surgery_mean
        self.cleaning_mean = cleaning_mean
        self.random_preparation = random_preparation
        self.random_surgery = random_surgery
        self.random_cleaning = random_cleaning

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

    def get_surgery_type(self):
        return self.surgery_type

    def get_release_date(self):
        return self.release_date

    def get_due_date(self):
        return self.due_date

    def get_priority(self):
        return self.priority

    def set_priority(self, priority):
        self.priority = priority

    def get_preparation_mean(self):
        return self.preparation_mean

    def get_surgery_mean(self):
        return self.surgery_mean

    def get_cleaning_mean(self):
        return self.cleaning_mean

    def get_random_preparation(self):
        return self.random_preparation

    def get_random_surgery(self):
        return self.random_surgery

    def get_random_cleaning(self):
        return self.random_cleaning

    def get_total_time(self):
        return self.get_preparation_time() + self.get_surgery_time() + self.get_cleaning_time()

    def __hash__(self):
        return hash(self.surgery_id)

    def __eq__(self, other):
        if not isinstance(other, Surgery):
            return NotImplemented
        return self.surgery_id == other.surgery_id

    def __lt__(self, other):
        if not isinstance(other, Surgery):
            return NotImplemented
        return self.surgery_id < other.surgery_id

    def __ne__(self, other):
        return not self.__eq__(other)

    def __le__(self, other):
        return self.__lt__(other) or self.__eq__(other)

    def __gt__(self, other):
        return not self.__le__(other)

    def __ge__(self, other):
        return not self.__lt__(other)
