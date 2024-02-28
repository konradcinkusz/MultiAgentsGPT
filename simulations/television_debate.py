import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import functools
from collections import OrderedDict
from typing import List

from langchain.schema import (
    HumanMessage,
    SystemMessage,
)
from simulators.dialogue_simulator import DialogueSimulator
from langchain_openai import ChatOpenAI
from agents.dialogue_agent import DialogueAgent
from agents.dialogue_agent_director import DirectorDialogueAgent
from simulations.interactions.television_debate.television_debate_description import TelevisionDebateDescription

topic = "The New Workout Trend: Competitive Sitting - How Laziness Became the Next Fitness Craze"
director_name = "Jon Stewart"

agent_summaries = OrderedDict(
    {
        "Jon Stewart": ("Host of the Daily Show", "New York"),
        "Samantha Bee": ("Hollywood Correspondent", "Los Angeles"),
        "Aasif Mandvi": ("CIA Correspondent", "Washington D.C."),
        "Ronny Chieng": ("Average American Correspondent", "Cleveland, Ohio"),
    }
)

word_limit = 50

television_debate_description = TelevisionDebateDescription(word_limit, topic, agent_summaries)

agent_descriptions = [
    television_debate_description.generate_description(name, role, location)
    for name, (role, location) in agent_summaries.items()
]
agent_headers = [
    television_debate_description.generate_agent_header(name, role, location, description)
    for (name, (role, location)), description in zip(
        agent_summaries.items(), agent_descriptions
    )
]
agent_system_messages = [
    television_debate_description.generate_system_message(name, header)
    for name, header in zip(agent_summaries, agent_headers)
]

# for name, description, header, system_message in zip(
#     agent_summaries, agent_descriptions, agent_headers, agent_system_messages
# ):
#     print(f"\n\n{name} Description:")
#     print(f"\n{description}")
#     print(f"\nHeader:\n{header}")
#     print(f"\nSystem Message:\n{system_message.content}")

topic_specifier_prompt = [
    SystemMessage(content="You can make a task more specific."),
    HumanMessage(
        content=f"""{television_debate_description.conversation_description}
        
        Please elaborate on the topic. 
        Frame the topic as a single question to be answered.
        Be creative and imaginative.
        Please reply with the specified topic in {word_limit} words or less. 
        Do not add anything else."""
    ),
]
specified_topic = ChatOpenAI(temperature=1.0)(topic_specifier_prompt).content

print(f"Original topic:\n{topic}\n")
print(f"Detailed topic:\n{specified_topic}\n")

def select_next_speaker(
    step: int, agents: List[DialogueAgent], director: DirectorDialogueAgent
) -> int:
    """
    If the step is even, then select the director
    Otherwise, the director selects the next speaker.
    """
    # the director speaks on odd steps
    if step % 2 == 1:
        idx = 0
    else:
        # here the director chooses the next speaker
        idx = director.select_next_speaker() + 1  # +1 because we excluded the director
    return idx


director = DirectorDialogueAgent(
    name=director_name,
    system_message=agent_system_messages[0],
    model=ChatOpenAI(temperature=0.2),
    speakers=[name for name in agent_summaries if name != director_name],
    stopping_probability=0.2,
)

agents = [director]
for name, system_message in zip(
    list(agent_summaries.keys())[1:], agent_system_messages[1:]
):
    agents.append(
        DialogueAgent(
            name=name,
            system_message=system_message,
            model=ChatOpenAI(temperature=0.2),
        )
    )

simulator = DialogueSimulator(
    agents=agents,
    selection_function=functools.partial(select_next_speaker, director=director),
)
simulator.reset()
simulator.inject("Audience member", specified_topic)
print(f"(Audience member): {specified_topic}")
print("\n")

# ANSI color codes
even_color = '\033[94m' # Blue for even iterations
odd_color = '\033[92m' # Green for odd iterations

n=0

while True:
    # Choose the color based on whether n is even or odd
    color = even_color if n % 2 == 0 else odd_color

    name, message = simulator.step()
    # Print the message in the selected color and reset color at the end
    print(f"{color}({name}): {message}\033[0m")
    print("\n")
    if director.stop or n > 10:
        break