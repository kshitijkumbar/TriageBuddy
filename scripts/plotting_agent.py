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

query = "Give just python code for loading the file 'test_file.csv' into a pandas dataframe, plot the columns 'is_fault' and plot the values and vertical lines of the column 'fault_code' for values where 'is_fault' is true. Plot these values versus the 'Time' column. Plot values in different colors"

# chat_with_repl = chat.bind_tools([repl_tool])
# msg = chat_with_repl.invoke(query)
# print(msg.tool_calls)
init_prompt = ChatPromptTemplate.from_template("You are a python code producing bot, Given the user query: {query}, Provide only the python code fo the query. Do not provide any other text")
correction_prompt = ChatPromptTemplate.from_template("You are a python code checking bot, Given the user code: {code}, Check code for correctness. If the code is incorrect, correct it and provide a corrected version. Only provide code, Do not provide any other text")
chain = init_prompt | chat |  StrOutputParser() | correction_prompt | chat | StrOutputParser() | repl_tool
answer = chain.invoke({"query":query})
print(answer)

# chain = prompt | chat | repl_tool
# answer = chain.invoke({"text": })
# print(answer)
