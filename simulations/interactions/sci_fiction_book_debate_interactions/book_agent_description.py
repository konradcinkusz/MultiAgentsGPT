from agent_interaction import AgentInteraction

from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

class BookAgentDescription(AgentInteraction):
    def __init__(self, word_limit):
        self.word_limit = word_limit
        self.agent_descriptor_system_message = SystemMessage(content="You can make a task more specific.")

    def generate_description(self, agent_name):
        agent_specifier_prompt = [
            self.agent_descriptor_system_message,
            HumanMessage(content=f"""
                Please reply with a creative description of the character, {agent_name}, in {self.word_limit} words or less. 
                Speak directly to {agent_name}.
                Do not add anything else."""
            ),
        ]
        character_description = ChatOpenAI(temperature=1.0)(
            agent_specifier_prompt
        ).content
        return character_description

    def generate_system_message(self, agent_name, role_agent, agent_description, observer):
        return SystemMessage(content=(
            f""". 
        Never forget you are the {role_agent}, {agent_name} 
        Your character description is as follows: {agent_description}.
        You will propose actions you plan to take and {observer} will explain what happens when you take those actions.
        Speak in the first person from the perspective of {agent_name}.
        You are going to propose your point of view about the given idea.
        Do not change roles!
        Do not add anything else.
        Stop speaking the moment you finish speaking from your perspective.
        """
            )
        )