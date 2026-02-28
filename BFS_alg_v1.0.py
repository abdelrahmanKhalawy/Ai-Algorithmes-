from collections import deque
from itertools import permutations

# All possible mappings
switches = ["Sa", "Sb", "Sc"]
bulbs = ["B1", "B2", "B3"]
all_mappings = list(permutations(bulbs))


# State = (Sa, Sb, Sc, warmed)
def initial_state():
    return (0, 0, 0, {})


def successors(state):
    Sa, Sb, Sc, warmed = state
    result = []

    # Toggle actions
    result.append(((1 - Sa, Sb, Sc, warmed.copy()), "Toggle Sa"))
    result.append(((Sa, 1 - Sb, Sc, warmed.copy()), "Toggle Sb"))
    result.append(((Sa, Sb, 1 - Sc, warmed.copy()), "Toggle Sc"))

    # Wait (warm bulbs controlled by ON switches)
    new_warm = warmed.copy()
    if Sa == 1:
        new_warm["Sa"] = True
    if Sb == 1:
        new_warm["Sb"] = True
    if Sc == 1:
        new_warm["Sc"] = True

    result.append(((Sa, Sb, Sc, new_warm), "Wait"))

    return result


def goal(state):
    Sa, Sb, Sc, warmed = state
    observations = {
        "Sa": ("ON" if Sa == 1 else ("WARM" if "Sa" in warmed else "COLD")),
        "Sb": ("ON" if Sb == 1 else ("WARM" if "Sb" in warmed else "COLD")),
        "Sc": ("ON" if Sc == 1 else ("WARM" if "Sc" in warmed else "COLD")),
    }
    if len(set(observations.values())) == 3:
        return True
    return False


def BFS():
    q = deque([(initial_state(), [], [])])  # state, path, actions
    visited = set()

    while q:
        state, path, actions = q.popleft()
        key = (state[0], state[1], state[2], tuple(sorted(state[3].items())))
        if key in visited:
            continue
        visited.add(key)

        if goal(state):
            return path + [state], actions, visited

        for s, action in successors(state):
            q.append((s, path + [state], actions + [action]))

    return None, None, visited


def describe_state(state):
    Sa, Sb, Sc, warmed = state
    desc = []
    for switch, val in zip(["Sa", "Sb", "Sc"], [Sa, Sb, Sc]):
        if val == 1:
            desc.append(f"{switch} is ON")
        elif switch in warmed:
            desc.append(f"{switch} is WARM")
        else:
            desc.append(f"{switch} is OFF/COLD")
    return ", ".join(desc)


# Run BFS
solution, actions, visited = BFS()

# Print user-friendly output
print("\n=== BFS Plan (User-Friendly) ===")
for i, (step, action) in enumerate(zip(solution, actions + ["Final State"])):
    print(f"Step {i + 1}: {describe_state(step)} | Action: {action}")

print(f"\nTotal nodes visited during BFS: {len(visited)}")
