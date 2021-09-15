import matplotlib.pyplot as plt

num_list = [0.13, 0.159, 0.530, 0.536]
names = ["Diagonal\n Scanning", "Random\n Exploration", "Seq length = 5", " Seq length = 10"]
plt.bar(names, num_list, color=['y', 'y', 'r', 'r'])
plt.ylim(0, 1)
plt.ylabel("Coverage Rate")

plt.show()
