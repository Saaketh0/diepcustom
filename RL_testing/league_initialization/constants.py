"""Shared policy IDs for main and ghost league policies."""

CHAR_CLASSES = ("A", "B", "C", "D")
GHOST_SLOTS = 4

MAIN_POLICIES = [f"main_class_{char_class}" for char_class in CHAR_CLASSES]
GHOST_POLICIES = [
    f"class_{char_class}_ghost_{slot}"
    for char_class in CHAR_CLASSES
    for slot in range(GHOST_SLOTS)
]


def main_policy_id(char_class: str) -> str:
    return f"main_class_{char_class}"


def ghost_policy_id(char_class: str, slot: int) -> str:
    return f"class_{char_class}_ghost_{int(slot)}"
