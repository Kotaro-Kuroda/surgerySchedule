class Surgeon:
    def __init__(self, surgeon_id, dict_group):
        self.surgeon_id = surgeon_id
        self.dict_group = dict_group

    def get_surgeon_id(self):
        return self.surgeon_id

    def get_dict_group(self):
        return self.dict_group

    def __hash__(self):
        return hash(self.surgeon_id)

    def __eq__(self, other):
        if not isinstance(other, Surgeon):
            return NotImplemented
        return self.surgeon_id == other.surgeon_id

    def __lt__(self, other):
        if not isinstance(other, Surgeon):
            return NotImplemented
        return self.surgeon_id < other.surgeon_id

    def __ne__(self, other):
        return not self.__eq__(other)

    def __le__(self, other):
        return self.__lt__(other) or self.__eq__(other)

    def __gt__(self, other):
        return not self.__le__(other)

    def __ge__(self, other):
        return not self.__lt__(other)
