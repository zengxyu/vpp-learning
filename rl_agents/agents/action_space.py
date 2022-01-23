from enum import IntEnum


class ActionsV0(IntEnum):
    FORCE_LEFT_5 = 0
    FORCE_LEFT_4 = 1
    FORCE_LEFT_3 = 2
    FORCE_LEFT_2 = 3
    FORCE_LEFT_1 = 4
    FORCE_FORWARD = 5
    FORCE_RIGHT_1 = 6
    FORCE_RIGHT_2 = 7
    FORCE_RIGHT_3 = 8
    FORCE_RIGHT_4 = 9
    FORCE_RIGHT_5 = 10


class ActionMapping:

    def __init__(self, action_enum):
        self.action_enum = action_enum

    def get_force(self, action):
        """action values are in self.action_enum"""
        force = action - sum(self.action_enum) / len(self.action_enum)
        return force

    def get_action_size(self):
        return len(self.action_enum)


if __name__ == '__main__':
    action_mapping = ActionMapping(ActionsV0)
    print(action_mapping.get_force(action=2))
    print(action_mapping.get_action_size())
