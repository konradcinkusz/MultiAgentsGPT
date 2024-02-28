import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from simulators.dialogue_simulator import DialogueSimulator
from simulators.select_alternately import select_next_speaker_alternately
from simulations.interactions.cars_research_interactions.cars_reasearch_interaction import CarsResearchInteraction
from agents.dialouge_agent_with_tools import DialogueAgentWithTools

from typing import Callable, List

from langchain.memory import ConversationBufferMemory
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI

#from dotenv import load_dotenv, find_dotenv
#load_dotenv(find_dotenv())
names = {
    "AI accelerationist": ["arxiv"],
    "AI alarmist": ["wikipedia"],
}
topic = "The current impact of automation and artificial intelligence on employment"
word_limit = 50  # word limit for task brainstorming


cars_research = CarsResearchInteraction(word_limit, topic, names)

agent_descriptions = {name: cars_research.generate_description(name) for name in names}
agent_system_messages = {
    name: cars_research.generate_system_message(name, description, tools)
    for (name, tools), description in zip(names.items(), agent_descriptions.values())
}

conversation_description = f"""Here is the topic of conversation: {topic}
The participants are: {', '.join(names.keys())}"""

agent_descriptor_system_message = SystemMessage(
    content="You can add detail to the description of the conversation participant."
)

topic_specifier_prompt = [
    SystemMessage(content="You can make a topic more specific."),
    HumanMessage(
        content=f"""{topic}
        
        You are the moderator.
        Please make the topic more specific.
        Please reply with the specified quest in {word_limit} words or less. 
        Speak directly to the participants: {*names,}.
        Do not add anything else."""
    ),
]
specified_topic = ChatOpenAI(temperature=1.0)(topic_specifier_prompt).content

print(f"Original topic:\n{topic}\n")
print(f"Detailed topic:\n{specified_topic}\n")

# we set `top_k_results`=2 as part of the `tool_kwargs` to prevent results from overflowing the context limit
agents = [
    DialogueAgentWithTools(
        name=name,
        system_message=SystemMessage(content=system_message),
        model=ChatOpenAI(model_name="gpt-4", temperature=0.2),
        tool_names=tools,
        top_k_results=2,
    )
    for (name, tools), system_message in zip(
        names.items(), agent_system_messages.values()
    )
]

max_iters = 6
n = 0

simulator = DialogueSimulator(agents=agents, selection_function=select_next_speaker_alternately)
simulator.reset()
simulator.inject("Moderator", specified_topic)
print(f"(Moderator): {specified_topic}")
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