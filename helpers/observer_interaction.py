from helpers.agent_interaction import AgentInteraction

from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

class ObserverInteraction(AgentInteraction):
    def __init__(self, word_limit):
        self.word_limit = word_limit
        self.agent_descriptor_system_message = SystemMessage(content="You can make a task more specific.")

    def generate_description(self, observer_name, debate_description="sci fiction book brainstorming"):
        observer_specifier_prompt = [
            self.agent_descriptor_system_message,
            HumanMessage(content=f"""{debate_description}
                Please reply with a detailed description of the observer, {observer_name}, in {self.word_limit} words or less. 
                Speak directly to {observer_name}.
                Focus on how {observer_name} should observe and report on the debate, including any specific qualities or abilities that will help them in this task.
                Do not add anything else."""
            ),
        ]
        observer_description = ChatOpenAI(temperature=1.0)(
            observer_specifier_prompt
        ).content
        return observer_description

    def generate_system_message(self, observer_name, observer_description, debate_description="sci fiction book brainstorming"):
        return SystemMessage(content=(
            f"""{debate_description}
        You are the observer, {observer_name}. 
        Your description is as follows: {observer_description}.
        As the debate unfolds, your role is to observe the arguments presented by each side, note the strengths and weaknesses, and provide unbiased summaries or insights.
        Speak in the third person from the perspective of {observer_name}, focusing on what you observe rather than participating in the debate.
        Do not take sides or express personal opinions on the arguments.
        Remember to keep your observations and summaries within {self.word_limit} words!
        Do not add anything else.
        """
            )
        )