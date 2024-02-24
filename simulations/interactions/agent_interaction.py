from abc import ABC, abstractmethod

class AgentInteraction(ABC):
    @abstractmethod
    def generate_description(self, name):
        pass

    @abstractmethod
    def generate_system_message(self, *args):
        pass


