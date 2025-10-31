import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from matplotlib.animation import FuncAnimation

# --- Manipulator parameters ---
l1 = 1.0  # length of link 1
l2 = 1.0  # length of link 2

# --- Workspace limits ---
x_lim = [-2, 2]
y_lim = [-2, 2]

# --- Start and goal end-effector positions ---
start_pos = np.array([0.5, 0.5])
goal_pos  = np.array([1.5, 1.0])
goal_tol = 0.05

# --- Obstacle definition (circular obstacle) ---
obstacle_center = np.array([1.0, 1.0])
obstacle_radius = 0.15

# --- Discretization parameters ---
N1 = 100  # number of bins for theta1 (e.g., 10 deg steps)
N2 = 100 # number of bins for theta2
theta1_vals = np.linspace(-np.pi, np.pi, N1, endpoint=False)  # wrap-around grid
theta2_vals = np.linspace(-np.pi, np.pi, N2, endpoint=False)


# Reward constants
REWARD_GOAL = 100.0
REWARD_COLLISION = -200.0
REWARD_STEP = -1.0

reward_grid = np.full((N1, N2), REWARD_STEP)



dtheta = np.deg2rad(10)  # 10° increments
actions = [
    (+dtheta, 0),
    (-dtheta, 0),
    (0, +dtheta),
    (0, -dtheta),
    (+dtheta, +dtheta),
    (-dtheta, -dtheta),
    (+dtheta, -dtheta),
    (-dtheta, +dtheta)
]
num_actions = len(actions)

def forward_kinematics(theta1, theta2):
    """Compute joint and end-effector positions for given angles."""
    x1 = l1 * np.cos(theta1)
    y1 = l1 * np.sin(theta1)
    x2 = x1 + l2 * np.cos(theta1 + theta2)
    y2 = y1 + l2 * np.sin(theta1 + theta2)
    return np.array([[0, x1, x2], [0, y1, y2]])

def plot_environment(theta1, theta2, ax=None):
    """Draw manipulator in workspace."""
    l1, l2 = 1.0, 1.0
    p0 = np.array([0, 0])
    p1 = np.array([l1*np.cos(theta1), l1*np.sin(theta1)])
    p2 = np.array([p1[0] + l2*np.cos(theta1 + theta2), p1[1] + l2*np.sin(theta1 + theta2)])
    
    if ax is None:
        fig, ax = plt.subplots()
    ax.clear()
    ax.set_aspect('equal')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_title("2R Manipulator Policy Execution")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    # Draw obstacle, start, goal
    circle = plt.Circle(obstacle_center, obstacle_radius, color='r', alpha=0.3)
    ax.add_patch(circle)
    ax.plot(start_pos[0], start_pos[1], 'go', label='Start')
    ax.plot(goal_pos[0], goal_pos[1], 'bo', label='Goal')

    # Draw links
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], 'k-', lw=3)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', lw=3)
    ax.plot(p2[0], p2[1], 'ko')
    ax.legend(loc='upper right')

    return ax

def check_collision(theta1, theta2, obstacle_center, obstacle_radius):
    """Return True if any link collides with circular obstacle."""
    pts = forward_kinematics(theta1, theta2)
    link1 = pts[:, 0:2].T  # [ [x0,y0], [x1,y1] ]
    link2 = pts[:, 1:3].T  # [ [x1,y1], [x2,y2] ]

    def segment_circle_collision(p1, p2, center, radius):
        p1, p2, c = np.array(p1), np.array(p2), np.array(center)
        v = p2 - p1
        w = c - p1
        t = np.dot(w, v) / np.dot(v, v)
        t = np.clip(t, 0.0, 1.0)  # clamp to segment
        closest = p1 + t * v
        dist = np.linalg.norm(closest - c)
        return dist <= radius

    # check both links
    coll1 = segment_circle_collision(link1[0], link1[1], obstacle_center, obstacle_radius)
    coll2 = segment_circle_collision(link2[0], link2[1], obstacle_center, obstacle_radius)

    return coll1 or coll2

# --- Map from (i,j) index to angle values ---
def get_next_state(i, j, action_idx):
    """Given discrete state indices (i,j) and action index, return next indices and collision flag."""
    th1 = theta1_vals[i]
    th2 = theta2_vals[j]
    d1, d2 = actions[action_idx]
    th1_new = (th1 + d1 + np.pi) % (2*np.pi) - np.pi
    th2_new = (th2 + d2 + np.pi) % (2*np.pi) - np.pi

    # find nearest discrete index for new angles
    i_new = (np.abs(theta1_vals - th1_new)).argmin()
    j_new = (np.abs(theta2_vals - th2_new)).argmin()

    coll = collision_grid[i_new, j_new]
    return i_new, j_new, coll



# --- evaluate grid ---
collision_grid = np.zeros((N1, N2), dtype=bool)
goal_grid = np.zeros((N1, N2), dtype=bool)
start_grid = np.zeros((N1, N2), dtype=bool)
ee_x = np.zeros((N1, N2))
ee_y = np.zeros((N1, N2))

for i, th1 in enumerate(theta1_vals):
    for j, th2 in enumerate(theta2_vals):
        pts = forward_kinematics(th1, th2)
        ee = np.array([pts[0,2], pts[1,2]])
        ee_x[i,j] = ee[0]
        ee_y[i,j] = ee[1]
        collision_grid[i,j] = check_collision(th1, th2, obstacle_center, obstacle_radius)
        goal_grid[i,j] = np.linalg.norm(ee - goal_pos) <= goal_tol
        start_grid[i,j] = np.linalg.norm(ee - start_pos) <= goal_tol


for i in range(N1):
    for j in range(N2):
        if collision_grid[i, j]:
            reward_grid[i, j] = REWARD_COLLISION
        elif goal_grid[i, j]:
            reward_grid[i, j] = REWARD_GOAL
        else:
            reward_grid[i, j] = REWARD_STEP

# --- Summary ---
total_states = N1 * N2
n_coll = np.count_nonzero(collision_grid)
n_goal = np.count_nonzero(goal_grid)
n_free = total_states - n_coll
print(f"Total states: {total_states}, Free: {n_free}, In-collision: {n_coll}, Goal-states: {n_goal}")

# --- Plot 1: Joint-space grid (theta1 vs theta2) ---
# We'll plot each grid cell as a colored square: white=free, red=collision, green=goal (overrides free)
# fig1 = plt.figure(figsize=(6,6))
# ax1 = fig1.add_subplot(111)
# ax1.set_title("Joint-space discretization (θ1 vs θ2)")
# ax1.set_xlabel("θ2 index")
# ax1.set_ylabel("θ1 index")
# ax1.set_xlim(-0.5, N2-0.5)
# ax1.set_ylim(-0.5, N1-0.5)
# ax1.set_xticks(np.linspace(0, N2-1, 9))
# ax1.set_yticks(np.linspace(0, N1-1, 9))
# ax1.grid(False)

# # draw cells
# for i in range(N1):
#     for j in range(N2):
#         if collision_grid[i,j]:
#             rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, color='red', alpha=0.9)
#             ax1.add_patch(rect)
#         else:
#             rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, color='white', edgecolor='lightgray', alpha=1.0)
#             ax1.add_patch(rect)
#         if goal_grid[i,j]:
#             rectg = plt.Rectangle((j-0.5, i-0.5), 1, 1, color='green', alpha=0.9)
#             ax1.add_patch(rectg)
#         if start_grid[i,j]:
#             rects = plt.Rectangle((j-0.5, i-0.5), 1, 1, color='blue', alpha=0.9)
#             ax1.add_patch(rects)

# ax1.invert_yaxis()  # so lower index appears at top (optional)
# plt.show()


# -------------------- Value Iteration --------------------
gamma = 0.95
theta = 1e-3  # convergence threshold
max_iters = 10000

V = np.zeros((N1, N2)) 

for i in range(N1):
    for j in range(N2):
        if collision_grid[i,j] or goal_grid[i,j]:
            V[i,j] = reward_grid[i,j]


iteration = 0
while True:
    delta = 0.0
    V_new = V.copy()
    for i in range(N1):
        for j in range(N2):
            if collision_grid[i,j] or goal_grid[i,j]:
                continue  # skip terminal (absorbing)
            best = -np.inf
            for a in range(num_actions):
                ni, nj, _ = get_next_state(i, j, a)
                r = reward_grid[ni, nj]  # reward of the resulting state (we chose reward on next state)
                # deterministic
                val = r + gamma * V[ni, nj]
                if val > best:
                    best = val
            V_new[i,j] = best
            delta = max(delta, abs(V_new[i,j] - V[i,j]))
    V = V_new
    iteration += 1
    if delta < theta or iteration >= max_iters:
        break

print(f"Value iteration converged in {iteration} iterations; delta={delta:.6f}")

# -------------------- Extract greedy policy --------------------
policy = -np.ones((N1, N2), dtype=int)  # action index per state
for i in range(N1):
    for j in range(N2):
        if collision_grid[i,j] or goal_grid[i,j]:
            policy[i,j] = -1  # no action (terminal)
        else:
            best = -np.inf; besta = 0
            for a in range(num_actions):
                ni, nj, _ = get_next_state(i, j, a)
                r = reward_grid[ni, nj]
                val = r + gamma * V[ni, nj]
                if val > best:
                    best = val; besta = a
            policy[i,j] = besta

# -------------------- Find start state index (nearest to start_pos) --------------------
# choose the state whose end-effector is closest to provided start_pos and is collision-free
start_idx = None
min_dist = np.inf
for i in range(N1):
    for j in range(N2):
        if collision_grid[i,j]:
            continue
        dist = np.linalg.norm(np.array([ee_x[i,j], ee_y[i,j]]) - start_pos)
        if dist < min_dist:
            min_dist = dist; start_idx = (i, j)
print(f"Start index chosen: {start_idx}, distance to start_pos={min_dist:.4f} m")


max_steps = 500
traj = []
cur = start_idx
reached = False
for step in range(max_steps):
    traj.append(cur)
    i, j = cur
    if goal_grid[i,j]:
        reached = True; break
    a = policy[i,j]
    if a == -1:
        break  # terminal or undefined

    ni, nj,_ = get_next_state(i, j, a)


    # if we get stuck in self-loop, stop
    if (ni, nj) == (i, j):
        break
    cur = (ni, nj)
traj = np.array(traj)

print(f"Trajectory length: {len(traj)}. Reached goal: {reached}")


# Convert trajectory indices to joint angles
theta1_traj = [theta1_vals[i] for i, j in traj]
theta2_traj = [theta2_vals[j] for i, j in traj]

fig, ax = plt.subplots(figsize=(6,6))
ax.set_aspect('equal')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_title("2R Manipulator MDP Path")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")

# Static elements
circle = plt.Circle(obstacle_center, obstacle_radius, color='r', alpha=0.3)
ax.add_patch(circle)
ax.plot(start_pos[0], start_pos[1], 'go', label='Start')
ax.plot(goal_pos[0], goal_pos[1], 'bo', label='Goal')
ax.legend()

# Moving parts
link1, = ax.plot([], [], 'k-', lw=3)
link2, = ax.plot([], [], 'k-', lw=3)
joint, = ax.plot([], [], 'ko', markersize=6)

def init():
    link1.set_data([], [])
    link2.set_data([], [])
    joint.set_data([], [])
    return link1, link2, joint

def update(frame):
    th1 = theta1_traj[frame]
    th2 = theta2_traj[frame]
    p0 = np.array([0, 0])
    p1 = np.array([l1*np.cos(th1), l1*np.sin(th1)])
    p2 = np.array([p1[0] + l2*np.cos(th1 + th2), p1[1] + l2*np.sin(th1 + th2)])
    
    link1.set_data([p0[0], p1[0]], [p0[1], p1[1]])
    link2.set_data([p1[0], p2[0]], [p1[1], p2[1]])
    joint.set_data(p2[0], p2[1])
    return link1, link2, joint

anim = FuncAnimation(fig, update, frames=len(traj), init_func=init,
                     interval=200, blit=True, repeat=False)

plt.show()



# # Example test:
# theta1 = np.deg2rad(30)
# theta2 = np.deg2rad(30)
# collision = check_collision(theta1, theta2, obstacle_center, obstacle_radius)
# print("Collision:", collision)

# # visualize manipulator and mark if collision
# plot_environment(theta1, theta2)
# if collision:
#     print("⚠️ Manipulator in collision with obstacle!")
# else:
#     print("✅ Manipulator is collision-free.")


# Example usage:
# plot_environment(theta1=np.deg2rad(0), theta2=np.deg2rad(0))
