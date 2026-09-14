from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SubagentRole:
    """A focused, read-only specialist used for delegated model work."""

    name: str
    description: str
    instructions: str


SUBAGENT_ROLES = {
    role.name: role
    for role in (
        SubagentRole(
            "researcher",
            "Investigates APIs, documentation, and implementation options",
            "Research the task carefully. Return evidence, tradeoffs, and a concise "
            "recommendation. Do not modify files.",
        ),
        SubagentRole(
            "coder",
            "Designs concrete implementation changes",
            "Act as a senior implementer. Inspect relevant code and return a precise "
            "implementation proposal, including files and edge cases. Do not modify "
            "files.",
        ),
        SubagentRole(
            "reviewer",
            "Finds correctness, security, and maintainability issues",
            "Review the requested area skeptically. Prioritize actionable findings by "
            "severity and explain how to fix them. Do not modify files.",
        ),
        SubagentRole(
            "tester",
            "Develops verification strategies and test cases",
            "Analyze the task as a test engineer. Return high-value test cases, "
            "failure modes, and exact verification commands. Do not modify files.",
        ),
    )
}


def build_subagent_prompt(role_name: str, task: str) -> tuple[SubagentRole, str]:
    role = SUBAGENT_ROLES.get(role_name.lower())
    if role is None:
        available = ", ".join(SUBAGENT_ROLES)
        raise ValueError(f"Unknown subagent role {role_name!r}; choose: {available}")
    task = task.strip()
    if not task:
        raise ValueError("Subagent task cannot be empty")
    prompt = (
        f"You are the {role.name} subagent for a larger AI-agent session.\n\n"
        f"Role instructions: {role.instructions}\n\n"
        f"Delegated task: {task}\n\n"
        "Return a self-contained report for the main agent and user."
    )
    return role, prompt
