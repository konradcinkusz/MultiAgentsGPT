from langchain_openai import ChatOpenAI
from agents.dialogue_agent import DialogueAgent

from langchain.schema import SystemMessage
from langchain.prompts import PromptTemplate

class BiddingDialogueAgent(DialogueAgent):
    def __init__(
        self,
        name,
        system_message: SystemMessage,
        bidding_template: PromptTemplate,
        model: ChatOpenAI,
    ) -> None:
        super().__init__(name, system_message, model)
        self.bidding_template = bidding_template

    def bid(self) -> str:
        """
        Asks the chat model to output a bid to speak
        """
        prompt = PromptTemplate(
            input_variables=["message_history", "recent_message"],
            template=self.bidding_template,
        ).format(
            message_history="\n".join(self.message_history),
            recent_message=self.message_history[-1],
        )
        bid_string = self.model([SystemMessage(content=prompt)]).content
        return bid_string