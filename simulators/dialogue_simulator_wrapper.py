from simulators.dialogue_simulator import DialogueSimulator


class DebateSimulatorWrapper:
    def __init__(self, agents, director, selection_function):
        self.simulator = DialogueSimulator(
            agents=agents,
            selection_function=selection_function,
        )
        self.director = director

    def run_simulation(self, specified_topic):
        self.simulator.reset()
        self.simulator.inject("Audience member", specified_topic)
        print(f"(Audience member): {specified_topic}")
        print("\n")

        n = 0
        while True:
            name, message = self.simulator.step()
            speaker = next(agent for agent in self.simulator.agents if agent.name == name)
            print(f"{speaker.color}({name}): {message}\033[0m")
            print("\n")
            if self.director.stop or n > 10:
                break
            n += 1