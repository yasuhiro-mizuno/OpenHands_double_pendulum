import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import odeint

# Saving flags
SAVE_GIF = False
SAVE_VIDEO = False
SAVE_PLOTS = False

class DoublePendulum:
    """
    Class representing a double pendulum physical model.
    """
    def __init__(self, L1=1.0, L2=1.0, m1=1.0, m2=1.0, g=9.81):
        self.L1 = L1
        self.L2 = L2
        self.m1 = m1
        self.m2 = m2
        self.g = g

    def derivatives(self, state, t):
        """
        Compute derivatives [dθ1, dθ2, dω1, dω2] at time t.
        """
        theta1, theta2, omega1, omega2 = state
        delta = theta2 - theta1
        m1, m2, L1, L2, g = self.m1, self.m2, self.L1, self.L2, self.g

        # Denominators
        denom1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta)**2
        denom2 = (L2 / L1) * denom1

        # Angular accelerations
        domega1 = (
            -m2 * L1 * omega1**2 * np.sin(delta) * np.cos(delta)
            + m2 * g * np.sin(theta2) * np.cos(delta)
            + m2 * L2 * omega2**2 * np.sin(delta)
            - (m1 + m2) * g * np.sin(theta1)
        ) / denom1

        domega2 = (
            -m2 * L2 * omega2**2 * np.sin(delta) * np.cos(delta)
            + (m1 + m2) * g * np.sin(theta1) * np.cos(delta)
            - (m1 + m2) * L1 * omega1**2 * np.sin(delta)
            - (m1 + m2) * g * np.sin(theta2)
        ) / denom2

        return [omega1, omega2, domega1, domega2]

    def simulate(self, initial_conditions, t_span, dt=0.01):
        """
        Simulate the double pendulum over time.

        initial_conditions: [θ1, θ2, ω1, ω2]
        t_span: tuple (t_start, t_end)
        dt: time step

        Returns:
            t: time array
            sol: array of state variables over time
        """
        t = np.arange(t_span[0], t_span[1] + dt, dt)
        sol = odeint(self.derivatives, initial_conditions, t)
        return t, sol

    def get_cartesian_coords(self, sol):
        """
        Convert solution (θ1, θ2) to cartesian coordinates.

        sol: array shape (N,4)

        Returns:
            x1, y1, x2, y2: arrays of positions
        """
        theta1 = sol[:, 0]
        theta2 = sol[:, 1]
        x1 = self.L1 * np.sin(theta1)
        y1 = -self.L1 * np.cos(theta1)
        x2 = x1 + self.L2 * np.sin(theta2)
        y2 = y1 - self.L2 * np.cos(theta2)
        return x1, y1, x2, y2


def create_animation(pendulum, t, sol, init_cond,
                     save_gif=SAVE_GIF, save_video=SAVE_VIDEO,
                     filename_prefix='animation'):
    """
    Create and display an animation for a single simulation.
    """
    x1, y1, x2, y2 = pendulum.get_cartesian_coords(sol)
    dt = t[1] - t[0]

    fig, ax = plt.subplots()
    max_len = pendulum.L1 + pendulum.L2
    ax.set_xlim(-max_len, max_len)
    ax.set_ylim(-max_len, max_len)
    ax.set_aspect('equal')
    ax.set_title(f"theta1={init_cond[0]:.2f}, theta2={init_cond[1]:.2f}")
    line, = ax.plot([], [], 'o-', lw=2)

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        thisx = [0, x1[i], x2[i]]
        thisy = [0, y1[i], y2[i]]
        line.set_data(thisx, thisy)
        return line,

    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=len(t), interval=dt * 1000, blit=True
    )

    plt.show()

    if save_gif:
        try:
            anim.save(f"{filename_prefix}.gif", writer='pillow', fps=int(1/dt))
        except Exception as e:
            print(f"Saving GIF failed: {e}, saving as HTML")
            anim.save(f"{filename_prefix}.html", writer='html')
    if save_video:
        try:
            anim.save(f"{filename_prefix}.mp4", writer='ffmpeg', fps=int(1/dt))
        except Exception as e:
            print(f"Saving MP4 failed: {e}, saving as HTML")
            anim.save(f"{filename_prefix}.html", writer='html')


def create_dual_animation(pendulum, t, sol1, sol2, init1, init2,
                          save_gif=SAVE_GIF, save_video=SAVE_VIDEO,
                          filename_prefix='dual_animation'):
    """
    Create and display side-by-side animations for two simulations.
    """
    x11, y11, x12, y12 = pendulum.get_cartesian_coords(sol1)
    x21, y21, x22, y22 = pendulum.get_cartesian_coords(sol2)
    dt = t[1] - t[0]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    max_len = pendulum.L1 + pendulum.L2
    for ax in axes:
        ax.set_xlim(-max_len, max_len)
        ax.set_ylim(-max_len, max_len)
        ax.set_aspect('equal')
    axes[0].set_title(f"θ1={init1[0]:.2f}, θ2={init1[1]:.2f}")
    axes[1].set_title(f"θ1={init2[0]:.2f}, θ2={init2[1]:.2f}")
    line1, = axes[0].plot([], [], 'o-', lw=2)
    line2, = axes[1].plot([], [], 'o-', lw=2)

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        return line1, line2

    def animate(i):
        line1.set_data([0, x11[i], x12[i]], [0, y11[i], y12[i]])
        line2.set_data([0, x21[i], x22[i]], [0, y21[i], y22[i]])
        return line1, line2

    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=len(t), interval=dt * 1000, blit=True
    )

    plt.show()

    if save_gif:
        try:
            anim.save(f"{filename_prefix}.gif", writer='pillow', fps=int(1/dt))
        except Exception as e:
            print(f"Saving GIF failed: {e}, saving as HTML")
            anim.save(f"{filename_prefix}.html", writer='html')
    if save_video:
        try:
            anim.save(f"{filename_prefix}.mp4", writer='ffmpeg', fps=int(1/dt))
        except Exception as e:
            print(f"Saving MP4 failed: {e}, saving as HTML")
            anim.save(f"{filename_prefix}.html", writer='html')


def plot_trajectories(pendulum, t, sol1, sol2, init1, init2,
                      save_plots=SAVE_PLOTS, filename='trajectories.png'):
    """
    Create and display a static plot comparing two simulations.
    """
    x11, y11, x12, y12 = pendulum.get_cartesian_coords(sol1)
    x21, y21, x22, y22 = pendulum.get_cartesian_coords(sol2)

    fig = plt.figure(figsize=(12, 4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.5])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    # Final pendulum positions
    ax1.plot([0, x11[-1], x12[-1]], [0, y11[-1], y12[-1]], 'o-', lw=2)
    ax1.set_title(f"θ1={init1[0]:.2f}, θ2={init1[1]:.2f}")
    ax1.set_aspect('equal')
    ax1.set_xlim(-pendulum.L1 - pendulum.L2, pendulum.L1 + pendulum.L2)
    ax1.set_ylim(-pendulum.L1 - pendulum.L2, pendulum.L1 + pendulum.L2)

    ax2.plot([0, x21[-1], x22[-1]], [0, y21[-1], y22[-1]], 'o-', lw=2)
    ax2.set_title(f"θ1={init2[0]:.2f}, θ2={init2[1]:.2f}")
    ax2.set_aspect('equal')
    ax2.set_xlim(-pendulum.L1 - pendulum.L2, pendulum.L1 + pendulum.L2)
    ax2.set_ylim(-pendulum.L1 - pendulum.L2, pendulum.L1 + pendulum.L2)

    # Trajectories in XY plane
    ax3.plot(x12, y12, label='Cond1')
    ax3.plot(x22, y22, label='Cond2')
    ax3.scatter([x12[-1], x22[-1]], [y12[-1], y22[-1]],
                color=['C0', 'C1'], marker='o')
    ax3.set_title('Tip Trajectories')
    ax3.set_xlabel('X [m]')
    ax3.set_ylabel('Y [m]')
    ax3.legend()
    ax3.set_aspect('equal')

    plt.tight_layout()
    plt.show()

    if save_plots:
        try:
            fig.savefig(filename)
        except Exception as e:
            print(f"Saving plot failed: {e}")


def main():
    # Create a double pendulum instance
    pendulum = DoublePendulum()
    # Define two sets of initial conditions ([θ1, θ2, ω1, ω2])
    init1 = [np.pi/2, np.pi/2, 0, 0]
    init2 = [np.pi/2 + 0.01, np.pi/2, 0, 0]
    # Simulate for 0 to 20 seconds with dt = 0.01
    t, sol1 = pendulum.simulate(init1, (0, 20), dt=0.01)
    _, sol2 = pendulum.simulate(init2, (0, 20), dt=0.01)

    # Create and show animations and plots
    create_dual_animation(pendulum, t, sol1, sol2, init1, init2)
    plot_trajectories(pendulum, t, sol1, sol2, init1, init2)


if __name__ == '__main__':
    main()
