import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from interactions.sci_fiction_book_debate_interactions.book_agent_description import BookAgentDescription
from interactions.sci_fiction_book_debate_interactions.observer_interaction import ObserverInteraction

from typing import Callable, List

from langchain.schema import (
    HumanMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI

from agents.dialogue_agent import DialogueAgent
from agents.dialogue_simulator import DialogueSimulator

observer_name = "Isaac Asimov"

engineer_name = "Henry Ford"
philosopher_name = "Aristotle"
sportman_name = "Robert Lewandowski"
businessman_name = "Warren Buffet"
president_name = "Joe Biden"
politic_name = "Donald Trump"

agent_names = [engineer_name, philosopher_name, sportman_name, businessman_name, president_name, politic_name]
agent_roles = ["engineer", "philosopher", "sportsman", "buisnessmann", "president", "politic"]
goal = "Think about the conspectus of the book related to Elon's Musk Neuralink. It should be futuristic story about near future."
word_limit = 30

book_agent_description = BookAgentDescription(word_limit)

agent_descriptions = [
    book_agent_description.generate_description(agent_name) for agent_name in agent_names
]
agent_system_messages = [
    book_agent_description.generate_system_message(agent_name, agent_role, agent_description, observer_name)
    for agent_name, agent_role, agent_description in zip(
        agent_names, agent_roles, agent_descriptions
    )
]

goal_specifier_prompt = [
    SystemMessage(content="You can make a task more specific."),
    HumanMessage(
        content=f"""
        The entry goal is enclosed withing triple ` below:
        ```
        {goal}
        ```
        And you have to follow the points:
        You are the creative person.
        Please make the goal more specific. Be creative and imaginative.
        Please reply with the specified quest in {word_limit} words or less. 
        Do not add anything else."""
    ),
]
specified_goal = ChatOpenAI(temperature=1.0)(goal_specifier_prompt).content

print(f"Original goal:\n{goal}\n")
print(f"Detailed goal:\n{specified_goal}\n")

agents = []
for agent_name, agent_system_message in zip(
    agent_names, agent_system_messages
):
    agents.append(
        DialogueAgent(
            name=agent_name,
            system_message=agent_system_message,
            model=ChatOpenAI(temperature=0.2),
        )
    )

observer_interaction = ObserverInteraction(word_limit)
observer_description = observer_interaction.generate_description(observer_name)
observer_system_message = observer_interaction.generate_system_message(observer_name, observer_description)

observer = DialogueAgent(
    name=observer_name,
    system_message=observer_system_message,
    model=ChatOpenAI(temperature=0.2),
)

def select_next_speaker(step: int, agents: List[DialogueAgent]) -> int:
    """
    If the step is even, then select the storyteller
    Otherwise, select the other characters in a round-robin fashion.

    For example, with three characters with indices: 1 2 3
    The storyteller is index 0.
    Then the selected index will be as follows:

    step: 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16

    idx:  0  1  0  2  0  3  0  1  0  2  0  3  0  1  0  2  0
    """
    if step % 2 == 0:
        idx = 0
    else:
        idx = (step // 2) % (len(agents) - 1) + 1
    return idx

max_iters = 7
n = 0

simulator = DialogueSimulator(
    agents=[observer] + agents, selection_function=select_next_speaker
)
simulator.reset()
simulator.inject(observer_name, specified_goal)
print(f"({observer_name}): {specified_goal}")
print("\n")

# ANSI color codes
even_color = '\033[94m' # Blue for even iterations
odd_color = '\033[92m' # Green for odd iterations

while n < max_iters:
    # Choose the color based on whether n is even or odd
    color = even_color if n % 2 == 0 else odd_color

    name, message = simulator.step()
    # Print the message in the selected color and reset color at the end
    print(f"{color}({name}): {message}\033[0m")
    print("\n")
    n += 1