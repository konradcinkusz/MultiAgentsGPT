import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulations.interactions.agent_interaction import AgentInteraction

from langchain.schema import (
    HumanMessage,
    SystemMessage,
)

from langchain_openai import ChatOpenAI

class TelevisionDebateDescription(AgentInteraction):
    def __init__(self, word_limit, topic, agent_summaries):
        self.word_limit = word_limit
        self.topic = topic
        self.agent_summaries = agent_summaries
        self._agent_summary_string = "\n- ".join(
    [""]
    + [
        f"{name}: {role}, located in {location}"
        for name, (role, location) in self.agent_summaries.items()
    ]
)
        self.conversation_description = f"""This is a Daily Show episode discussing the following topic: {topic}.

The episode features {self._agent_summary_string}."""

        self._agent_descriptor_system_message = SystemMessage(
    content="You can add detail to the description of each person."
)

    def generate_description(self, agent_name, agent_role, agent_location):
        agent_specifier_prompt = [
            self._agent_descriptor_system_message,
            HumanMessage(
                content=f"""{self.conversation_description}
                Please reply with a creative description of {agent_name}, who is a {agent_role} in {agent_location}, that emphasizes their particular role and location.
                Speak directly to {agent_name} in {self.word_limit} words or less.
                Do not add anything else."""
            ),
        ]
        agent_description = ChatOpenAI(temperature=1.0)(agent_specifier_prompt).content
        return agent_description

    def generate_system_message(self, agent_name, agent_header):
        return SystemMessage(
            content=(
                f"""{agent_header}
    You will speak in the style of {agent_name}, and exaggerate your personality.
    Do not say the same things over and over again.
    Speak in the first person from the perspective of {agent_name}
    For describing your own body movements, wrap your description in '*'.
    Do not change roles!
    Do not speak from the perspective of anyone else.
    Speak only from the perspective of {agent_name}.
    Stop speaking the moment you finish speaking from your perspective.
    Never forget to keep your response to {self.word_limit} words!
    Do not add anything else.
        """
            )
        )

    def generate_agent_header(self, agent_name, agent_role, agent_location, agent_description):
        return f"""{self.conversation_description}

    Your name is {agent_name}, your role is {agent_role}, and you are located in {agent_location}.

    Your description is as follows: {agent_description}

    You are discussing the topic: {self.topic}.

    Your goal is to provide the most informative, creative, and novel perspectives of the topic from the perspective of your role and your location.
    """