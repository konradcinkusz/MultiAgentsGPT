from langchain.output_parsers import RegexParser

class IntegerOutputParser(RegexParser):
    def get_format_instructions(self) -> str:
        return "You have to response only with an integer number. Remember do not create introduction or summarization. Return only integer number as a result."
        #return "Your response should be an integer delimited by angled brackets, like this: <int>."
