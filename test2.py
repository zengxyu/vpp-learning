episodes = 50000
start_ratio = 0.05
end_ratio = 0.0005
for i in range(0, episodes):
    ratio = max(start_ratio - 0.0001 * i, end_ratio)
    print("第i={}次:{}".format(i, ratio))
