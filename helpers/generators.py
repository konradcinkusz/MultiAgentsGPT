from langchain.schema import (
    HumanMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI

agent_descriptor_system_message = SystemMessage(content="You can make a task more specific.")
word_limit = 30

def generate_agent_description(agent_name):
    agent_specifier_prompt = [
        agent_descriptor_system_message,
        HumanMessage(
            content=f"""
            Please reply with a creative description of the character, {agent_name}, in {word_limit} words or less. 
            Speak directly to {agent_name}.
            Do not add anything else."""
        ),
    ]
    character_description = ChatOpenAI(temperature=1.0)(
        agent_specifier_prompt
    ).content
    return character_description

def generate_agent_system_message(agent_name, role_agent, agent_description, observer):
    return SystemMessage(
        content=(
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

def generate_observer_description(observer_name, debate_description = "sci fiction book brainstorming"):
    observer_specifier_prompt = [
        agent_descriptor_system_message,
        HumanMessage(
            content=f"""{debate_description}
            Please reply with a detailed description of the observer, {observer_name}, in {word_limit} words or less. 
            Speak directly to {observer_name}.
            Focus on how {observer_name} should observe and report on the debate, including any specific qualities or abilities that will help them in this task.
            Do not add anything else."""
        ),
    ]
    observer_description = ChatOpenAI(temperature=1.0)(
        observer_specifier_prompt
    ).content
    return observer_description

def generate_observer_system_message(observer_name, observer_description, debate_description = "sci fiction book brainstorming"):
    return SystemMessage(
        content=(
            f"""{debate_description}
    You are the observer, {observer_name}. 
    Your description is as follows: {observer_description}.
    As the debate unfolds, your role is to observe the arguments presented by each side, note the strengths and weaknesses, and provide unbiased summaries or insights.
    Speak in the third person from the perspective of {observer_name}, focusing on what you observe rather than participating in the debate.
    Do not take sides or express personal opinions on the arguments.
    Remember to keep your observations and summaries within {word_limit} words!
    Do not add anything else.
    """
        )
    )