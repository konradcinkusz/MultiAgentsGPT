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

from simulators.dialogue_simulator_wrapper import DebateSimulatorWrapper
from simulators.dialogue_simulator import DialogueSimulator
from langchain_openai import ChatOpenAI
from agents.dialogue_agent import DialogueAgent
from agents.dialogue_agent_director import DirectorDialogueAgent
from simulations.interactions.television_debate.television_debate_description import TelevisionDebateDescription

from langchain_community.document_loaders import PyPDFLoader
topic = "Debate about basics of ML"
director_name = "Andrew NG"

agent_summaries = OrderedDict(
    {
        "Andrew NG": ("Host of the Daily Show", "New York"),
        "Geoffrey Hinton": ("Pioneer in Neural Networks and Deep Learning", "Toronto"),
        "Yann LeCun": ("Chief AI Scientist at a major social media company", "New York"),
        "Fei-Fei Li": ("Leading expert in computer vision and AI ethics", "Stanford"),
        "Demis Hassabis": ("Founder of an AI research company focused on AGI", "London"),
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
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
assets_dir = os.path.join(base_dir, 'assets')

# https://github.com/ksm26/LangChain-Chat-with-Your-Data/blob/main/L1-Document_loading.ipynb
# https://see.stanford.edu/materials/aimlcs229/transcripts/MachineLearning-Lecture01.pdf
# https://see.stanford.edu/materials/aimlcs229/transcripts/MachineLearning-Lecture02.pdf
loaders = [
    PyPDFLoader(os.path.join(assets_dir, "MachineLearning-Lecture01.pdf")),
    PyPDFLoader(os.path.join(assets_dir, "MachineLearning-Lecture01.pdf")),
    PyPDFLoader(os.path.join(assets_dir, "MachineLearning-Lecture02.pdf")),
    PyPDFLoader(os.path.join(assets_dir, "MachineLearning-Lecture03.pdf"))
]
director.persist_directory = 'docs/chroma/'
director.create_vector_store(loaders)

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

# Replace the main loop with the new wrapper class
debate_simulator_wrapper = DebateSimulatorWrapper(
    agents=agents,
    director=director,
    selection_function=functools.partial(select_next_speaker, director=director),
)

debate_simulator_wrapper.run_simulation(specified_topic)