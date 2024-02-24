import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.dialogue_agent_bidding import BiddingDialogueAgent
from agents.dialogue_simulator import DialogueSimulator
from simulations.interactions.presidental_debate.bid_output_parser import BidOutputParser
from simulations.interactions.presidental_debate.presidental_debate_description import PresidentialDebateDescription

from langchain.schema import (
    HumanMessage,
    SystemMessage,
)

from langchain_openai import ChatOpenAI

debate_members_names = ["Donald Trump", "Kanye West", "Elizabeth Warren"]
topic = "transcontinental high speed rail"
word_limit = 50

debate_member_interactions = PresidentialDebateDescription(word_limit, topic, debate_members_names)

debate_members_descriptions = [
    debate_member_interactions.generate_description(debate_member_name) for debate_member_name in debate_members_names
]

debate_members_headers = [
    debate_member_interactions.generate_debate_member_header(debate_member_name, debate_member_description)
    for debate_member_name, debate_member_description in zip(
        debate_members_names, debate_members_descriptions
    )
]

debate_members_system_messages = [
    debate_member_interactions.generate_system_message(debate_member_name, debate_member_headers)
    for debate_member_name, debate_member_headers in zip(debate_members_names, debate_members_headers)
]

for (
    debate_member_name,
    debate_member_description,
    debate_member_header,
    debate_member_system_message,
) in zip(
    debate_members_names,
    debate_members_descriptions,
    debate_members_headers,
    debate_members_system_messages,
):
    print(f"\n\n{debate_member_name} Description:")
    print(f"\n{debate_member_description}")
    print(f"\n{debate_member_header}")
    print(f"\n{debate_member_system_message.content}")

debate_members_bidding_templates = [
    debate_member_interactions.generate_character_bidding_template(debate_member_header)
    for debate_member_header in debate_members_headers
]

for debate_member_name, bidding_template in zip(
    debate_members_names, debate_members_bidding_templates
):
    print(f"{debate_member_name} Bidding Template:")
    print(bidding_template)

topic_specifier_prompt = [
    SystemMessage(content="You can make a task more specific."),
    HumanMessage(
        content=f"""{debate_member_interactions.debate_description}
        
        You are the debate moderator.
        Please make the debate topic more specific. 
        Frame the debate topic as a problem to be solved.
        Be creative and imaginative.
        Please reply with the specified topic in {word_limit} words or less. 
        Speak directly to the presidential candidates: {*debate_members_names,}.
        Do not add anything else."""
    ),
]

specified_topic = ChatOpenAI(temperature=1.0)(topic_specifier_prompt).content

print(f"Original topic:\n{topic}\n")
print(f"Detailed topic:\n{specified_topic}\n")


members = []
for character_name, character_system_message, bidding_template in zip(
    debate_members_names, debate_members_system_messages, debate_members_bidding_templates
):
    members.append(
        BiddingDialogueAgent(
            name=character_name,
            system_message=character_system_message,
            model=ChatOpenAI(temperature=0.2),
            bidding_template=bidding_template,
        )
    )

max_iters = 10
n = 0

simulator = DialogueSimulator(agents=members, selection_function=debate_member_interactions.select_next_speaker)
simulator.reset()
simulator.inject("Debate Moderator", specified_topic)
print(f"(Debate Moderator): {specified_topic}")
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