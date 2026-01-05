import numpy as np
import matplotlib.pyplot as plt

# Set style for publication quality
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'text.usetex': False, # Using simple text for speed/compatibility
    'figure.figsize': (8, 5)
})

def generate_abcd_figure():
    # Define phase range
    phi = np.linspace(-np.pi, 2.5*np.pi, 500)
    
    # Define intensity signal: I = 1 + cos(phi)
    intensity = 1 + np.cos(phi)
    
    fig, ax = plt.subplots()
    
    # Plot the full fringe
    ax.plot(phi, intensity, 'k-', linewidth=1.5, label='Intensity Fringe $I(\phi)$')
    
    # Define the "current" phase to measure
    current_phi = np.pi / 4  # Arbitrary phase to illustrate
    
    # The MMI outputs measure this fringe at specific offsets relative to the current phase
    # If the MMI is ideal, biases are 0, pi/2, pi, 3pi/2
    # So for a given input phase 'current_phi', the outputs see:
    # A = I(current_phi + 0)
    # B = I(current_phi + pi/2)
    # C = I(current_phi + pi)
    # D = I(current_phi + 3pi/2)
    
    # Actually, the "scan" usually moves the fringe across the detectors.
    # The ABCD method works by having 4 detectors effectively sampling 4 points 
    # separated by pi/2 *simultaneously* for a single momentary phase delay.
    # So we plot the 4 points at x = current_phi + offset
    
    offsets = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
    labels = ['A (0)', 'B ($\pi/2$)', 'C ($\pi$)', 'D ($3\pi/2$)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']
    
    points_x = current_phi + offsets
    points_y = 1 + np.cos(points_x)
    
    for x, y, label, color, marker in zip(points_x, points_y, labels, colors, markers):
        ax.plot(x, y, color=color, marker=marker, markersize=10, linestyle='None', label=f'Output {label}')
        # Draw drop lines
        ax.vlines(x, 0, y, color=color, linestyle='--', alpha=0.5)
        
    # Annotate the phase being measured
    ax.annotate(r'Unknown Phase $\phi$', 
                xy=(current_phi, 0), xytext=(current_phi, -0.3),
                arrowprops=dict(arrowstyle='->', color='black'),
                ha='center')

    # Formatting
    ax.set_xlabel('Phase Delay (radians)')
    ax.set_ylabel('Normalized Intensity')
    ax.set_title('Principe du Système ABCD Intégré')
    
    # Clean up x-axis ticks
    ticks = np.arange(-np.pi, 3*np.pi, np.pi/2)
    tick_labels = [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$', r'$5\pi/2$']
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    
    ax.set_ylim(-0.5, 2.5)
    ax.set_xlim(-np.pi/2, 2.5*np.pi)
    
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('abcd_principle.png', dpi=300)
    # plt.savefig('abcd_principle.pdf') # Optional
    print("Figure generated: abcd_principle.png")

if __name__ == "__main__":
    generate_abcd_figure()
