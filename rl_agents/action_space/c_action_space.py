class DiscreteActionSpace:
    def __init__(self, n):
        self.n = n
        self.action_ind = [i for i in range(n)]

    def to_force(self, action_ind):
        avg_ind = sum(self.action_ind) / len(self.action_ind)
        return action_ind - avg_ind

    def __len__(self):
        return len(self.action_ind)


if __name__ == '__main__':
    das = DiscreteActionSpace(11)
    print(das.to_force(3))
