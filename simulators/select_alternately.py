from typing import List
from agents.dialogue_agent import DialogueAgent

def select_next_speaker_alternately(step: int, agents: List[DialogueAgent]) -> int:
    idx = (step) % len(agents)
    return idx