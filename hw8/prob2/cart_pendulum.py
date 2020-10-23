from scipy.integrate import solve_ivp
from scipy.signal import place_poles

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation

class CartPend:
  def __init__(self, m, M, L, g, b):
    self.m = m
    self.M = M
    self.L = L
    self.g = g
    self.b = b

    # Linearized Pendulum System
    A = np.array(
    [[0, 1, 0, 0],
     [0, -b/M, -m*g/M, 0],
     [0, 0, 0, 1],
     [0, b/(M*L), (m+M)*g/(M*L), 0]])

    B = np.reshape([0, 1/M, 0, 1/(M*L)], (4, 1))

    self.A = A
    self.B = B

  def response(self, x0, start, end, n, K=np.zeros(4).T, r=np.zeros(4), linear=False):
    if linear:
      return solve_ivp(lambda t,y: self.linear_pend(t,y,K,r), (start, end), x0, t_eval=np.linspace(start,end,n))
    else:
      return solve_ivp(lambda t,y: self.nonlinear_pend(t,y,K,r), (start, end), x0, t_eval=np.linspace(start,end,n))

  def linear_pend(self, t, x, K, r):
    u = -K.dot(x-r)
    return self.A.dot(x) + self.B.dot(u)

  def nonlinear_pend(self, t, x, K, r):
    u = -K.dot(x-r)
    m,M,L,g,b = self.m, self.M, self.L, self.g, self.b

    sin = np.sin(x[2])
    cos = np.cos(x[2])
    D = m*L*L*(M+m*sin**2)

    f1 = x[1]
    f2 = (1/D)*(-m**2*L**2*g*sin*cos + m*L*L*(m*L*x[3]**2*sin - b*x[1])) + m*L*L*(1/D)*u
    f3 = x[3]
    f4 = (1/D)*((m+M)*m*g*L*sin - m*L*cos*(m*L*x[3]**2*sin - b*x[1])) + m*L*cos*(1/D)*u + 0.01 * np.random.normal(0,1)
    return f1,f2,f3,f4

  @staticmethod
  def plot_response(sol):
    plt.figure(figsize=(10,7))
    plt.subplot(221)
    plt.plot(sol.t,sol.y[0],'m',lw=2)
    plt.legend([r'$y$'],loc=1)
    plt.ylabel('y')
    plt.xlabel('Time (s)')
    plt.title('Position (m)')
    plt.xlim(0,10)

    plt.subplot(222)
    plt.plot(sol.t,sol.y[1],'g',lw=2)
    plt.legend([r'$v$'],loc=1)
    plt.ylabel('v')
    plt.xlabel('Time (s)')
    plt.title('Velocity (m/s)')
    plt.xlim(0,10)

    plt.subplot(223)
    plt.plot(sol.t,sol.y[2],'r',lw=2)
    plt.legend([r'$\theta$'],loc=1)
    plt.ylabel(r'$\theta$')
    plt.xlabel('Time (s)')
    plt.title('Angle (rad)')
    plt.xlim(0,10)

    plt.subplot(224)
    plt.plot(sol.t,sol.y[3],'y',lw=2)
    plt.legend([r'$\omega$',r'$q$'],loc=1)
    plt.xlabel('Time (s)')
    plt.ylabel(r'$\omega$')
    plt.title('Angular Velocity (rad/s)')
    plt.xlim(0,10)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,2))

    plt.tight_layout
    plt.subplots_adjust(left=1,right=2, top=1, wspace=0.2, hspace=0.35);


  # Source: https://apmonitor.com/do/index.php/Main/InvertedPendulum
  def simulate(self, K, r, start, end, ylim=(-0.8,2), n=200):
    # n = 200
    # tf = 20.0
    # m.time = np.linspace(0,tf,n)
    tf = end-start
    sol = self.response(np.zeros(4), start, end, n, K=K, r=r, linear=False)
    time = np.linspace(0, tf, n)

    y = sol.y[0]
    v = sol.y[1]
    theta = sol.y[2]
    omega = sol.y[3]

    x1 = y
    y1 = np.zeros(n)

    #suppose that l = 1
    x2 = 1*np.sin(theta)+x1
    x2b = 1.05*np.sin(theta)+x1
    y2 = 1*np.cos(theta)-y1
    y2b = 1.05*np.cos(theta)-y1

    fig = plt.figure(figsize=(8,6.4))
    ax = fig.add_subplot(111,autoscale_on=False,\
                         xlim=ylim,ylim=(-0.4,1.2))
    ax.set_xlabel('position')
    ax.get_yaxis().set_visible(False)

    crane_rail, = ax.plot([ylim[0]-0.4,ylim[1]+0.4],[-0.2,-0.2],'k-',lw=4)
    start, = ax.plot([0,0],[-0.5,1.5],'k:',lw=2)
    objective, = ax.plot([1,1],[-0.5,1.5],'k:',lw=2)
    mass1, = ax.plot([],[],linestyle='None',marker='s',\
                     markersize=40,markeredgecolor='k',\
                     color='orange',markeredgewidth=2)
    mass2, = ax.plot([],[],linestyle='None',marker='o',\
                     markersize=20,markeredgecolor='k',\
                     color='orange',markeredgewidth=2)
    line, = ax.plot([],[],'o-',color='orange',lw=4,\
                    markersize=6,markeredgecolor='k',\
                    markerfacecolor='k')
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05,0.9,'',transform=ax.transAxes)
    wgt_template = 'weight = %.1f'
    wgt_text = ax.text(0.75,0.9,'',transform=ax.transAxes)
    start_text = ax.text(-0.1,-0.3,'start',ha='right')
    end_text = ax.text(1.1,-0.3,'objective',ha='left')

    def init():
        mass1.set_data([],[])
        mass2.set_data([],[])
        line.set_data([],[])
        time_text.set_text('')
        wgt_text.set_text('')
        return line, mass1, mass2, time_text, wgt_text

    def animate(i):
        mass1.set_data([x1[i]],[y1[i]-0.1])
        mass2.set_data([x2b[i]],[y2b[i]])
        line.set_data([x1[i],x2[i]],[y1[i],y2[i]])
        time_text.set_text(time_template % time[i])
        return line, mass1, mass2, time_text

    ani_a = animation.FuncAnimation(fig,animate,np.arange(1,n),interval=30,blit=False,init_func=init)
    plt.close(fig);

    return ani_a
