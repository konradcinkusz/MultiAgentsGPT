import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tenacity
from agent_interaction import AgentInteraction
from typing import Callable, List
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from agents.dialogue_agent import DialogueAgent
import numpy as np

from simulations.interactions.presidental_debate.bid_output_parser import BidOutputParser

class PresidentialDebateDescription(AgentInteraction):
    def __init__(self, word_limit, topic, debate_members):
        self.word_limit = word_limit
        self.topic = topic
        self.agent_descriptor_system_message = SystemMessage(content="You can add detail to the description of each presidential candidate.")
        self.debate_description = f"""Here is the topic for the presidential debate: {topic}.
                        The presidential candidates are: {', '.join(debate_members)}."""
        self.bid_parser = BidOutputParser(
            regex=r"<(\d+)>", output_keys=["bid"], default_output_key="bid"
        )
    def generate_description(self, agent_name):
        character_specifier_prompt = [
            self.agent_descriptor_system_message,
            HumanMessage(
                content=f"""{self.debate_description}
                Please reply with a creative description of the presidential candidate, {agent_name}, in {self.word_limit} words or less, that emphasizes their personalities. 
                Speak directly to {agent_name}.
                Do not add anything else."""
            ),
        ]
        character_description = ChatOpenAI(temperature=1.0)(
            character_specifier_prompt
        ).content
        return character_description

    def generate_system_message(self, character_name, character_header):
        return SystemMessage(
            content=(
                f"""{character_header}
    You will speak in the style of {character_name}, and exaggerate their personality.
    You will come up with creative ideas related to {self.topic}.
    Do not say the same things over and over again.
    Speak in the first person from the perspective of {character_name}
    For describing your own body movements, wrap your description in '*'.
    Do not change roles!
    Do not speak from the perspective of anyone else.
    Speak only from the perspective of {character_name}.
    Stop speaking the moment you finish speaking from your perspective.
    Never forget to keep your response to {self.word_limit} words!
    Do not add anything else.
        """
            )
        )
    
    def generate_debate_member_header(self, character_name, character_description):
        return f"""{self.debate_description}
        Your name is {character_name}.
        You are a presidential candidate.
        Your description is as follows: {character_description}
        You are debating the topic: {self.topic}.
        Your goal is to be as creative as possible and make the voters think you are the best candidate.
        """
    
    def generate_character_bidding_template(self, character_header):
        
        bidding_template = f"""{character_header}

    ```
    {{message_history}}
    ```

    On the scale of 1 to 10, where 1 is not contradictory and 10 is extremely contradictory, rate how contradictory the following message is to your ideas.

    ```
    {{recent_message}}
    ```

    {self.bid_parser.get_format_instructions()}
    Do nothing else.
        """
        return bidding_template
    
    
    @tenacity.retry(
        stop=tenacity.stop_after_attempt(2),
        wait=tenacity.wait_none(),  # No waiting time between retries
        retry=tenacity.retry_if_exception_type(ValueError),
        before_sleep=lambda retry_state: print(
            f"ValueError occurred: {retry_state.outcome.exception()}, retrying..."
        ),
        retry_error_callback=lambda retry_state: 0,
    )  # Default value when all retries are exhausted
    def ask_for_bid(self, agent) -> str:
        """
        Ask for agent bid and parses the bid into the correct format.
        """
        bid_string = agent.bid()
        bid = int(self.bid_parser.parse(bid_string)["bid"])
        return bid
    
    def select_next_speaker(self, step: int, agents: List[DialogueAgent]) -> int:
        bids = []
        for agent in agents:
            bid = self.ask_for_bid(agent)
            bids.append(bid)

        # randomly select among multiple agents with the same bid
        max_value = np.max(bids)
        max_indices = np.where(bids == max_value)[0]
        idx = np.random.choice(max_indices)

        print("Bids:")
        for i, (bid, agent) in enumerate(zip(bids, agents)):
            print(f"\t{agent.name} bid: {bid}")
            if i == idx:
                selected_name = agent.name
        print(f"Selected: {selected_name}")
        print("\n")
        return idx