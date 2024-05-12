from langchain.agents import Tool
from langchain_experimental.utilities import PythonREPL
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.schema.output_parser import StrOutputParser

from operator import itemgetter

from dotenv import load_dotenv
import os
load_dotenv()




chat = ChatGroq(temperature=0, model_name="llama3-8b-8192")
system = "You are a helpful assistant."
human = "{text}"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])




python_repl = PythonREPL()


# You can create the tool to pass to an agent
repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python_repl.run,
)

query = "Give just python code for plotting 'Column1': [1, 2, 1, 4, 5] and 'Column2': [2, 3, 2, 5, 6] vs 'time' : [0, 1, 2, 3, 4]. Highlight any inflection points for a give time value for Column1 and Column2 in different colors."

# chat_with_repl = chat.bind_tools([repl_tool])
# msg = chat_with_repl.invoke(query)
# print(msg.tool_calls)
init_prompt = ChatPromptTemplate.from_template("You are a python code producing bot, Given the user query: {query}, Provide only the python code fo the query. Do not provide any other text")
correction_prompt = ChatPromptTemplate.from_template("You are a python code checking bot, Given the user code: {code}, Check code for correctness. If the code is incorrect, correct it and provide a corrected version. Do not provide any other text")
chain = init_prompt | chat |  StrOutputParser() | correction_prompt | chat | StrOutputParser() | repl_tool
answer = chain.invoke({"query":query})
print(answer)

# chain = prompt | chat | repl_tool
# answer = chain.invoke({"text": })
# print(answer)
