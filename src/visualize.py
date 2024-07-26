import numpy as np

from matplotlib import pyplot as plt

def visualize(solutions, 
              link_lengths, 
              objectives, 
              ax, 
              context):
    """Plots multiple arms with the given angles and link lengths on ax.
    
    Args:
        solutions (list of np.ndarray): A list of (dim,) arrays with the joint angles of the arms.
        link_lengths (np.ndarray): The length of each link the arm.
        objectives (list of float): The objectives of these solutions.
        ax (plt.Axes): A matplotlib axis on which to display the arms.
    """
    lim = 1.05 * np.sum(link_lengths)  # Add a bit of a border.
    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    
    for i, solution in enumerate(solutions):
        ax.set_title(f"Objective: {objectives[i]}")
        pos = np.array([0, 0])  # Starting position of the next joint.
        cum_thetas = np.cumsum(solution)
        for link_length, cum_theta in zip(link_lengths, cum_thetas):
            # Calculate the end of this link.
            next_pos = pos + link_length * np.array(
                [np.cos(cum_theta), np.sin(cum_theta)])
            ax.plot([pos[0], next_pos[0]], [pos[1], next_pos[1]], "-ko", ms=0.5, linewidth=0.5)
            pos = next_pos

        # Add points for the start and end positions and conditioned positions
        ax.plot(context[0, 0].item(), context[0, 1].item(), "cx", ms=10)
        ax.plot(0, 0, "ro", ms=2)
        final_label = f"Final: ({pos[0]:.2f}, {pos[1]:.2f})"
        ax.plot(pos[0], pos[1], "go", ms=2, label=final_label)
