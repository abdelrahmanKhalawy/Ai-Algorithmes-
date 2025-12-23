from heapq import heappush, heappop
from itertools import count as itertools_count


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

    # Wait action (warm bulbs controlled by ON switches)
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
    return len(set(observations.values())) == 3


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


def heuristic(state):
    # عدد switches اللي لسه مش مختلفة عن بعض في الحالة النهائية
    Sa, Sb, Sc, warmed = state
    obs = [
        "ON" if Sa == 1 else ("WARM" if "Sa" in warmed else "COLD"),
        "ON" if Sb == 1 else ("WARM" if "Sb" in warmed else "COLD"),
        "ON" if Sc == 1 else ("WARM" if "Sc" in warmed else "COLD"),
    ]
    return 3 - len(set(obs))  # عدد القيم المتكررة => تقدير للخطوات المتبقية


def A_star():
    counter = itertools_count()
    q = []
    heappush(
        q, (0 + heuristic(initial_state()), 0, next(counter), initial_state(), [], [])
    )
    visited = set()

    while q:
        f, g, _, state, path, actions = heappop(q)
        key = (state[0], state[1], state[2], tuple(sorted(state[3].items())))
        if key in visited:
            continue
        visited.add(key)

        if goal(state):
            return path + [state], actions, visited

        for s, action in successors(state):
            g_new = g + 1  # كل خطوة تكلفتها 1
            f_new = g_new + heuristic(s)
            heappush(
                q, (f_new, g_new, next(counter), s, path + [state], actions + [action])
            )

    return None, None, visited


# Run A*
solution, actions, visited = A_star()

# Print user-friendly output
print("\n=== A* Plan (User-Friendly) ===")
for i, (step, action) in enumerate(zip(solution, actions + ["Final State"])):
    print(f"Step {i + 1}: {describe_state(step)} | Action: {action}")

print(f"\nTotal nodes visited during A*: {len(visited)}")
