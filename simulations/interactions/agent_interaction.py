from abc import ABC, abstractmethod

class AgentInteraction(ABC):
    @abstractmethod
    def generate_description(self, *args):
        pass

    @abstractmethod
    def generate_system_message(self, *args):
        pass


