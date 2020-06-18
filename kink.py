import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
plt.style.use('seaborn-pastel')


mu=1
lmbda=1
x=np.linspace(-20,20,101)
a=np.linspace(-30,30,101)
b=np.linspace(-15,15,101)
u=0.5
t=np.linspace(0,100,101)

kink = lambda x: (mu/np.sqrt(lmbda))*np.tanh(mu*x/np.sqrt(2))
antikink = lambda x:-1*kink(x)
soliton = lambda x: 4*np.arctan(np.exp(x))
movesoliton= lambda x:4*np.arctan(np.exp(x-u*t)/np.sqrt(1-u**2))

plt.title("Graph kink") # заголовок
plt.xlabel("x") # ось абсцисс
plt.ylabel("kink") # ось ординат
plt.grid()      # включение отображение сетки
plt.plot(x, kink(x)+antikink(x-a))


plt.title("soliton") # заголовок
plt.xlabel("x") # ось абсцисс
plt.ylabel("kink") # ось ординат
plt.grid()      # включение отображение сетки
plt.plot(x, soliton(x-b))


plt.title("move soliton") # заголовок
plt.xlabel("x") # ось абсцисс
plt.ylabel("kink") # ось ординат
plt.grid()      # включение отображение сетки
plt.plot(x, movesoliton(x+5)+movesoliton(x-5))

plt.title("") # заголовок
plt.xlabel("x") # ось абсцисс
plt.ylabel("kink") # ось ординат
plt.grid()      # включение отображение сетки
plt.plot(x, soliton((x+5-u*t))/np.sqrt(1-u**2)+soliton((-x-5-u*t))/np.sqrt(1-u**2))

fig = plt.figure()
ax = plt.axes(xlim=(-20, 20), ylim=(-2, 2))
plt.title('kink+antikink')
line, = ax.plot([], [], lw=3)


Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

def init():
    line.set_data([], [])
    return line,


def kink_antikink(i):
    x = np.linspace(-20, 20, 101)
    y = kink(x)+antikink(x-(i-50))
    line.set_data(x, y)
    return line,


anim = FuncAnimation(fig, kink_antikink, init_func=init,
                     frames=100, interval=100, blit=True)
anim.save('kink_antikink.mp4', writer=writer)

