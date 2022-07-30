import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
xs = []
ys = []
yy = []

# Initialize communication with TMP102

# This function is called periodically from FuncAnimation
def animate(i, xs, ys, yy):

    # Read temperature (Celsius) from TMP102
    temp_c = np.random.uniform(0, 2000)/2

    # Add x and y to lists
    xs.append(i)
    ys.append(temp_c)
    yy.append(max(set(ys) | set(yy)))

    # Limit x and y lists to 20 items
    xs = xs[-20:]
    ys = ys[-20:]
    yy = yy[-20:]

    # Draw x and y lists
    ax.clear()
    ax.plot(xs, ys, 'b-', linewidth=0.7)
    ax.plot(xs, yy, 'r--', linewidth =0.7)

    # Format plot

# Set up plot to call animate() function periodically
best = 0
ani = animation.FuncAnimation(fig, animate, fargs=(xs, ys, yy))
plt.show()