# search_tree_compact_v2.py
"""
Compact Tree Visualizer V2
- Compact hierarchical layout (subtree widths)
- Zoom & Pan, canvas auto-resize
- Integrates BFS / UCS / A* from BFS_alg.py, UCS_alg.py, A_star_alg.py
- Full labels: "S:010" and "W:Sa,Sb"
- Coloring: unvisited(dashed), frontier(gray), expanded(orange), dead-end(red), solution(green)
- Minimal controls: Algorithm select, Prepare Search, Next Step
"""

import tkinter as tk
from tkinter import ttk
from collections import deque
from heapq import heappush, heappop
from itertools import count as itertools_count

# ---------------- Import user algorithms (must be in same folder) ----------------
import BFS_alg as bfs_mod
import UCS_alg as ucs_mod
import A_star_alg as astar_mod

initial_state_fn = bfs_mod.initial_state
successors_fn = bfs_mod.successors
goal_fn = bfs_mod.goal
heuristic_fn = astar_mod.heuristic  # only used by A*


# ---------------- State key helpers ----------------
def state_to_key(state):
    Sa, Sb, Sc, warmed = state
    if isinstance(warmed, dict):
        warmed_keys = tuple(sorted(warmed.keys()))
    elif isinstance(warmed, (list, tuple, set)):
        warmed_keys = tuple(sorted(warmed))
    else:
        warmed_keys = tuple()
    return (Sa, Sb, Sc, warmed_keys)


def key_label(key):
    Sa, Sb, Sc, warmed_keys = key
    wk = ",".join(warmed_keys) if warmed_keys else "cold"
    return f"S:{Sa}{Sb}{Sc}\nW:{wk}"


# ---------------- Build graph ----------------
def build_tree_bfs(max_nodes=1000):
    start = initial_state_fn()
    start_k = state_to_key(start)
    q = deque([start])
    visited = {start_k}
    nodes = [start_k]
    edges = []

    while q and len(nodes) < max_nodes:
        s = q.popleft()
        sk = state_to_key(s)

        for succ, action in successors_fn(s):
            sk2 = state_to_key(succ)

            # إذا ظهر قبل كده → ما تربطوش تاني
            if sk2 in visited:
                continue

            visited.add(sk2)
            nodes.append(sk2)
            edges.append((sk, sk2, action))
            q.append(succ)

    return nodes, edges


# ---------------- Search wrappers ----------------
def bfs_search_record():
    start = initial_state_fn()
    q = deque([(start, [])])
    visited = set()
    order = []
    parent = {}
    action_from_parent = {}
    while q:
        state, path = q.popleft()
        key = state_to_key(state)
        if key in visited:
            continue
        visited.add(key)
        order.append(key)
        if goal_fn(state):
            sol = path + [key]
            return {
                "visited": visited,
                "order": order,
                "solution": sol,
                "parent": parent,
                "action": action_from_parent,
            }
        for succ, action in successors_fn(state):
            sk = state_to_key(succ)
            if sk not in visited and sk not in parent:
                parent[sk] = key
                action_from_parent[sk] = action
            q.append((succ, path + [key]))
    return {
        "visited": visited,
        "order": order,
        "solution": None,
        "parent": parent,
        "action": action_from_parent,
    }


def ucs_search_record():
    counter = itertools_count()
    start = initial_state_fn()
    q = []
    heappush(q, (0, next(counter), start, []))
    visited = set()
    order = []
    parent = {}
    action_from_parent = {}
    while q:
        cost, _, state, path = heappop(q)
        key = state_to_key(state)
        if key in visited:
            continue
        visited.add(key)
        order.append(key)
        if goal_fn(state):
            sol = path + [key]
            return {
                "visited": visited,
                "order": order,
                "solution": sol,
                "parent": parent,
                "action": action_from_parent,
            }
        for succ, action in successors_fn(state):
            sk = state_to_key(succ)
            if sk not in visited:
                heappush(q, (cost + 1, next(counter), succ, path + [key]))
                if sk not in parent:
                    parent[sk] = key
                    action_from_parent[sk] = action
    return {
        "visited": visited,
        "order": order,
        "solution": None,
        "parent": parent,
        "action": action_from_parent,
    }


def astar_search_record():
    counter = itertools_count()
    start = initial_state_fn()
    q = []
    gstart = 0
    fstart = gstart + heuristic_fn(start)
    heappush(q, (fstart, gstart, next(counter), start, []))
    visited = set()
    order = []
    parent = {}
    action_from_parent = {}
    while q:
        f, g, _, state, path = heappop(q)
        key = state_to_key(state)
        if key in visited:
            continue
        visited.add(key)
        order.append(key)
        if goal_fn(state):
            sol = path + [key]
            return {
                "visited": visited,
                "order": order,
                "solution": sol,
                "parent": parent,
                "action": action_from_parent,
            }
        for succ, action in successors_fn(state):
            sk = state_to_key(succ)
            if sk not in visited:
                gnew = g + 1
                fnew = gnew + heuristic_fn(succ)
                heappush(q, (fnew, gnew, next(counter), succ, path + [key]))
                if sk not in parent:
                    parent[sk] = key
                    action_from_parent[sk] = action
    return {
        "visited": visited,
        "order": order,
        "solution": None,
        "parent": parent,
        "action": action_from_parent,
    }


# ---------------- Compact tree layout helpers ----------------
def build_children_map(nodes, edges):
    cm = {k: [] for k in nodes}
    for a, b, action in edges:
        if a in cm:
            cm[a].append(b)
    return cm


def compute_subtree_sizes(root, children_map):
    """
    Return dict size[node] = number of leaves under node (or weight)
    Use 1 for leaf, else sum of children sizes.
    """
    size = {}
    visited = set()

    def dfs(u):
        if u in size:
            return size[u]
        ch = children_map.get(u, [])
        if not ch:
            size[u] = 1.0
            return size[u]
        s = 0.0
        for v in ch:
            s += dfs(v)
        size[u] = max(s, 1.0)
        return size[u]

    dfs(root)
    return size


def assign_compact_positions(root, children_map, level_gap=140, node_gap=60):
    """
    Compute x,y per node using subtree widths (compact tidy layout).
    Returns positions dict node->(x,y)
    """
    sizes = compute_subtree_sizes(root, children_map)
    positions = {}
    # keep track of current x cursor
    cursor = {"x": 0.0}

    def place(u, depth):
        ch = children_map.get(u, [])
        y = depth * level_gap
        if not ch:
            x = cursor["x"]
            positions[u] = (x, y)
            cursor["x"] += node_gap
            return
        # place children first
        for v in ch:
            place(v, depth + 1)
        # center parent above its children
        first = positions[ch[0]][0]
        last = positions[ch[-1]][0]
        mid = (first + last) / 2.0
        positions[u] = (mid, y)

    place(root, 0)
    # recenter x to center around 0
    xs = [p[0] for p in positions.values()]
    if xs:
        mean_x = (min(xs) + max(xs)) / 2.0
    else:
        mean_x = 0.0
    for k, (x, y) in positions.items():
        positions[k] = (x - mean_x, y)
    return positions


# ---------------- Drawing primitives ----------------
def draw_node(canvas, x, y, r, label, status):
    """
    status: "unvisited" (dashed), "frontier" (light gray), "expanded" (orange),
            "dead" (red), "solution" (green)
    """
    if status == "expanded":
        fill = "#f4a261"
        outline = "#000000"
        dash = None
    elif status == "frontier":
        fill = "#d8d8d8"
        outline = "#000000"
        dash = None
    elif status == "dead":
        fill = "#ff4d4d"
        outline = "#7f0000"
        dash = None
    elif status == "solution":
        fill = "#2ecc71"
        outline = "#145a32"
        dash = None
    else:
        fill = "#ffffff"
        outline = "#999999"
        dash = (4, 4)
    oval = canvas.create_oval(
        x - r, y - r, x + r, y + r, fill=fill, outline=outline, width=2, dash=dash
    )
    text = canvas.create_text(x, y, text=label, font=("Arial", 9, "bold"))
    return oval, text


# ---------------- Visualizer class ----------------
class CompactTreeVisualizer:
    def __init__(self, root):
        self.root = root
        root.title("Search Tree Visualizer (Compact V2)")

        # Canvas auto-resize
        self.canvas = tk.Canvas(root, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda e: self.redraw())

        # Controls (minimal)
        ctrl = tk.Frame(root)
        ctrl.pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=6)
        ttk.Label(ctrl, text="Algorithm:").pack(side=tk.LEFT)
        self.algo_var = tk.StringVar(value="BFS")
        self.algo_box = ttk.Combobox(
            ctrl, textvariable=self.algo_var, values=["BFS", "UCS", "A*"], width=8
        )
        self.algo_box.pack(side=tk.LEFT, padx=6)
        self.prepare_btn = ttk.Button(
            ctrl, text="Prepare Search", command=self.prepare_search
        )
        self.prepare_btn.pack(side=tk.LEFT, padx=6)
        self.next_btn = ttk.Button(
            ctrl, text="Next Step", command=self.next_step, state=tk.DISABLED
        )
        self.next_btn.pack(side=tk.LEFT, padx=20)
        self.status_label = ttk.Label(ctrl, text="Status: idle")
        self.status_label.pack(side=tk.RIGHT, padx=6)

        # Zoom & Pan vars
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.dragging = False
        self.last_mouse = (0, 0)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)  # Windows
        self.canvas.bind("<Button-4>", self.on_mouse_wheel_unix)  # Some Linux
        self.canvas.bind("<Button-5>", self.on_mouse_wheel_unix)

        # Graph data placeholders
        self.nodes = []
        self.edges = []
        self.children_map = {}
        self.positions = {}
        self.node_items = {}
        self.edge_items = []

        # Search/run data
        self.visit_order = []
        self.visit_idx = 0
        self.parent = {}
        self.action_map = {}
        self.solution_keys = None
        self.steps_ready = False

        # drawing params
        self.node_radius = 26
        self.level_gap = 160
        self.node_gap = 70

        # initial build
        self.build_graph_and_layout()

    # ---------- mouse handlers ----------
    def on_mouse_down(self, event):
        # if clicking on control area? we still pan
        self.dragging = True
        self.last_mouse = (event.x, event.y)

    def on_mouse_move(self, event):
        if self.dragging:
            dx = event.x - self.last_mouse[0]
            dy = event.y - self.last_mouse[1]
            self.offset_x += dx
            self.offset_y += dy
            self.last_mouse = (event.x, event.y)
            self.redraw()

    def on_mouse_up(self, event):
        self.dragging = False

    def on_mouse_wheel_unix(self, event):
        if event.num == 4:
            self.zoom_at(event.x, event.y, 1.12)
        elif event.num == 5:
            self.zoom_at(event.x, event.y, 1 / 1.12)

    def on_mouse_wheel(self, event):
        if event.delta > 0:
            self.zoom_at(event.x, event.y, 1.12)
        else:
            self.zoom_at(event.x, event.y, 1 / 1.12)

    def zoom_at(self, sx, sy, factor):
        wx, wy = self.screen_to_world(sx, sy)
        self.scale *= factor
        sx2, sy2 = self.world_to_screen(wx, wy)
        self.offset_x += sx - sx2
        self.offset_y += sy - sy2
        self.redraw()

    # ---------- transforms ----------
    def world_to_screen(self, wx, wy):
        w = int(self.canvas.winfo_width() or 800)
        h = int(self.canvas.winfo_height() or 600)
        sx = wx * self.scale + w / 2 + self.offset_x
        sy = wy * self.scale + h / 2 + self.offset_y
        return sx, sy

    def screen_to_world(self, sx, sy):
        w = int(self.canvas.winfo_width() or 800)
        h = int(self.canvas.winfo_height() or 600)
        wx = (sx - w / 2 - self.offset_x) / (self.scale + 1e-12)
        wy = (sy - h / 2 - self.offset_y) / (self.scale + 1e-12)
        return wx, wy

    # ---------- build graph & layout ----------
    def build_graph_and_layout(self):
        self.status_label.config(text="Building graph...")
        self.root_update()
        self.nodes, self.edges = build_tree_bfs(max_nodes=500)
        # children map
        self.children_map = build_children_map(self.nodes, self.edges)
        # find root = initial state key
        root_key = state_to_key(initial_state_fn())
        # compact positions
        self.positions = assign_compact_positions(
            root_key,
            self.children_map,
            level_gap=self.level_gap,
            node_gap=self.node_gap,
        )
        # create draw items
        self.create_draw_items()
        self.status_label.config(
            text=f"Graph: {len(self.nodes)} nodes, {len(self.edges)} edges"
        )

    def create_draw_items(self):
        self.canvas.delete("all")
        self.node_items.clear()
        self.edge_items.clear()
        # draw edges (light color)
        for a, b, action in self.edges:
            if a in self.positions and b in self.positions:
                ax, ay = self.world_to_screen(*self.positions[a])
                bx, by = self.world_to_screen(*self.positions[b])
                lid = self.canvas.create_line(ax, ay, bx, by, fill="#dddddd", width=2)
                self.edge_items.append((lid, a, b))
        # draw nodes
        for k in self.nodes:
            if k in self.positions:
                x, y = self.world_to_screen(*self.positions[k])
                oval, text = draw_node(
                    self.canvas, x, y, self.node_radius, key_label(k), "unvisited"
                )
                self.node_items[k] = (oval, text)
        # text area top-right for step label
        self.canvas.create_text(
            self.canvas.winfo_width() - 20,
            20,
            text="",
            anchor="ne",
            font=("Arial", 11),
            tags="step_text",
        )

    def redraw(self):
        # update edges positions
        for lid, a, b in self.edge_items:
            if a in self.positions and b in self.positions:
                ax, ay = self.world_to_screen(*self.positions[a])
                bx, by = self.world_to_screen(*self.positions[b])
                self.canvas.coords(lid, ax, ay, bx, by)
        # update nodes positions
        for k, (oval, txt) in self.node_items.items():
            if k in self.positions:
                x, y = self.world_to_screen(*self.positions[k])
                r = self.node_radius
                self.canvas.coords(oval, x - r, y - r, x + r, y + r)
                self.canvas.coords(txt, x, y)
        # update step text (keeps tag)
        self.root_update()

    def root_update(self):
        try:
            self.root.update_idletasks()
        except:
            pass

    # ---------- prepare search ----------
    def prepare_search(self):
        algo = self.algo_var.get()
        self.status_label.config(text=f"Running {algo} (recording)...")
        self.root_update()
        if algo == "BFS":
            res = bfs_search_record()
        elif algo == "UCS":
            res = ucs_search_record()
        else:
            res = astar_search_record()
        self.visit_order = res["order"]
        self.parent = res["parent"]
        self.action_map = res["action"]
        self.solution_keys = res["solution"]
        self.visit_idx = 0
        self.steps_ready = True
        # reset visuals
        self.reset_visuals()
        self.status_label.config(
            text=f"{algo} prepared: visited {len(res['visited'])} nodes"
        )
        if self.visit_order:
            self.next_btn.config(state=tk.NORMAL)
        else:
            self.next_btn.config(state=tk.DISABLED)

    def reset_visuals(self):
        # reset node styles
        for k, (oval, txt) in self.node_items.items():
            self.canvas.itemconfig(
                oval, fill="#ffffff", outline="#999999", dash=(4, 4), width=2
            )
            self.canvas.itemconfig(txt, text=key_label(k))
        for lid, a, b in self.edge_items:
            self.canvas.itemconfig(lid, fill="#dddddd", width=2)
        self.canvas.delete("pointer")
        self.canvas.itemconfig("step_text", text="")

    # ---------- utility: dead-end detection ----------
    def is_dead_end(self, key, visited_up_to_idx):
        # outgoing edges
        outgoing = [b for (a, b, _) in self.edges if a == key]
        if not outgoing:
            return True
        # if all outgoing are already visited at or before visited_up_to_idx -> dead
        for b in outgoing:
            if b not in self.visit_order[: visited_up_to_idx + 1]:
                return False
        return True

    # ---------- next step ----------
    def next_step(self):
        if not self.steps_ready or self.visit_idx >= len(self.visit_order):
            self.status_label.config(
                text="No more steps or press Prepare Search first."
            )
            self.next_btn.config(state=tk.DISABLED)
            if self.solution_keys:
                self.mark_solution(self.solution_keys)
            return

        key = self.visit_order[self.visit_idx]

        # mark expanded (orange)
        if key in self.node_items:
            oval, txt = self.node_items[key]
            self.canvas.itemconfig(
                oval, fill="#f4a261", outline="#000000", dash=(), width=3
            )

        # mark children as frontier (light gray) if not yet visited
        for a, b, action in self.edges:
            if a == key and b in self.node_items:
                if b not in self.visit_order[: self.visit_idx + 1]:
                    oval_c, txt_c = self.node_items[b]
                    self.canvas.itemconfig(
                        oval_c, fill="#d8d8d8", outline="#000000", dash=(), width=2
                    )

        # detect dead end
        dead = self.is_dead_end(key, self.visit_idx)
        if dead and key in self.node_items:
            oval, txt = self.node_items[key]
            self.canvas.itemconfig(oval, fill="#ff4d4d", outline="#7f0000", width=3)

        # pointer (remove old then draw)
        self.canvas.delete("pointer")
        if key in self.positions:
            sx, sy = self.world_to_screen(*self.positions[key])
            # left of node
            self.canvas.create_polygon(
                sx - (self.node_radius + 18),
                sy,
                sx - (self.node_radius + 6),
                sy - 8,
                sx - (self.node_radius + 6),
                sy + 8,
                fill="#2aa198",
                outline="",
                tags="pointer",
            )

        # update step label
        action = self.action_map.get(key, "")
        step_txt = (
            f"Step {self.visit_idx + 1}/{len(self.visit_order)}  Action: {action}"
        )
        self.canvas.itemconfig("step_text", text=step_txt)

        # if key in solution, color green (we color all solution nodes eventually)
        if self.solution_keys and key in self.solution_keys:
            if key in self.node_items:
                oval_s, txt_s = self.node_items[key]
                self.canvas.itemconfig(
                    oval_s, fill="#2ecc71", outline="#145a32", width=3
                )

        self.visit_idx += 1

        # if reached goal node (last in solution), mark full solution and disable next button
        if self.solution_keys and key == self.solution_keys[-1]:
            self.mark_solution(self.solution_keys)
            self.status_label.config(text="Goal reached - solution highlighted.")
            self.next_btn.config(state=tk.DISABLED)
            return

    # ---------- mark solution nodes green ----------
    def mark_solution(self, sol_keys):
        for k in sol_keys:
            if k in self.node_items:
                oval, txt = self.node_items[k]
                self.canvas.itemconfig(oval, fill="#2ecc71", outline="#145a32", width=4)
        # remove pointer
        self.canvas.delete("pointer")


# ---------------- Run the app ----------------
if __name__ == "__main__":
    root = tk.Tk()
    app = CompactTreeVisualizer(root)
    root.mainloop()
