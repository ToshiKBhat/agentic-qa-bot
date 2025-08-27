from __future__ import annotations

class Hitl:
    """Simplest synchronous HITL adapter.
    mode: 'cli' prompts in terminal; 'auto' returns defaults.
    """
    def __init__(self, mode: str = "cli", auto_approve: bool = False):
        self.mode = mode
        self.auto_approve = auto_approve

    def ask(self, question: str, default: str = "") -> str:
        if self.mode != "cli" or self.auto_approve:
            return default or "OK"
        print("\n[HITL] " + question)
        try:
            ans = input("> ").strip()
        except EOFError:
            ans = ""
        return ans or default

    def confirm(self, prompt: str, default_yes: bool = True) -> bool:
        if self.mode != "cli" or self.auto_approve:
            return default_yes
        print(f"\n[HITL] {prompt} (y/n)")
        try:
            ans = input("> ").strip().lower()
        except EOFError:
            ans = ""
        if not ans:
            return default_yes
        return ans.startswith("y")