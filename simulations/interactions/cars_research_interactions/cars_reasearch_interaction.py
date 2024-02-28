import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent_interaction import AgentInteraction
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

class CarsResearchInteraction(AgentInteraction):
    def __init__(self, word_limit, topic, names):
        self.word_limit = word_limit
        self.topic = topic
        self.agent_descriptor_system_message = SystemMessage(content="You can add detail to the description of the conversation participant.")
        self.conversation_description = f"""Here is the topic of conversation: {topic}
            The participants are: {', '.join(names.keys())}"""
    
    def generate_description(self, name):
        agent_specifier_prompt = [
            self.agent_descriptor_system_message,
            HumanMessage(
                content=f"""{self.conversation_description}
                Please reply with a creative description of {name}, in {self.word_limit} words or less. 
                Speak directly to {name}.
                Give them a point of view.
                Do not add anything else."""
            ),
        ]
        agent_description = ChatOpenAI(temperature=1.0)(agent_specifier_prompt).content
        return agent_description

    def generate_system_message(self, name, description, tools):
        return f"""{self.conversation_description}
                
            Your name is {name}.

            Your description is as follows: {description}

            Your goal is to persuade your conversation partner of your point of view.

            DO look up information with your tool to refute your partner's claims.
            DO cite your sources.

            DO NOT fabricate fake citations.
            DO NOT cite any source that you did not look up.

            Do not add anything else.

            Stop speaking the moment you finish speaking from your perspective.
            """
    