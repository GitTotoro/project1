import matplotlib.pyplot as plt

plt.scatter(df['tavg'], df['total'], alpha=0.5)
plt.xlabel("Temperature")
plt.ylabel("Sales")
plt.title("Temperature vs Sales")
plt.show()
