from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMessage
from langchain_openai import ChatOpenAI
from langchain import hub
from typing import List
import os
from dotenv import load_dotenv

class RetailAgent:
    """
    AI agent that uses CRUD and EDA to answer questions about the retail dataset. Includes
    LangChain conversational memory.
    """
    def __init__(self, crud_obj, eda_obj):
        """
        :param crud_obj: instantiated CRUD object
        :param eda_obj: instantiated EDAFeatures object
        """
        load_dotenv()
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY")
        )

        # Preserve last 10 messages

        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=10
        )

        self.tools = [
            crud_obj.sales_by_year_tool,
            crud_obj.sales_by_month_tool,
            crud_obj.sales_by_date_tool,
            eda_obj.sales_t_tool,
            eda_obj.holiday_impact_tool,
            eda_obj.top_performing_stores_tool,
            eda_obj.department_analysis_holiday_tool,
            eda_obj.department_analysis_no_holiday_tool,
            eda_obj.analyze_economic_headwinds_tool,
            eda_obj.analyze_event_impact_tool,
            eda_obj.forecast_sales_tool,
            eda_obj.forecast_sales_faster_tool
        ]

        prompt = hub.pull("hwchase17/openai-tools-agent")
        enhanced_instructions = """
You are RetailMind, an expert AI business consultant specialized in retail analytics. 
Your goal is to provide concise, data-driven analysis and strategic advice to retail decision-makers.

IMPORTANT ERROR HANDLING INSTRUCTIONS:
- If any tool returns -1, this means the required data is not loaded or available
- When you encounter a -1 return value:
  1. Inform the user about the data availability issue
  2. Suggest they check the data loading process
  3. Provide alternative analysis approaches if possible
- Always check data availability before performing complex analyses
- If data cannot be loaded, clearly explain to the user what data is missing

CONVERSATION GUIDELINES:
- Reference previous questions and answers when relevant
- Build upon earlier analysis in the conversation
- Provide consistent, coherent responses across the session
- Remember user preferences and focus areas mentioned earlier
- When a user's query includes keywords like 'impact', 'effect', 'analyze', or 'change' related to may be an event (e.g., 'inventory changes', 'marketing campaign'), your FIRST action should be to proactively use the event_impact_analyzer tool with the potential event name. If the tool returns and says the event was not found, then inform the user the specific event isn't in the data and ask for clarification or suggest alternative analyses.
- When you are unsure about the date range or any specifics of a user's query, but you are able to associate it with a function, invoke that function anyway with a general parameter or a broad range, retrying at most 2 times until you get a successful response. Then, and only then, if after 3 tries you cannot formulate a response, ask the user to clarify their query."""
        prompt.messages[0].prompt.template += enhanced_instructions

        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        self.executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            return_intermediate_steps=True
        )

    def ask(self, question: str) -> str:
        """
        Asks the agent a question and returns an answer.
        Uses LangChain's built-in memory for conversation context.
        :param question (str): The user's question in natural language.
        :return: answer
        """
        try:
            response = self.executor.invoke({
                "input": question,
                "chat_history": self.memory.chat_memory.messages
            })
            return response['output']
        except Exception as e:
            return f"I encountered an error while processing your request:\n{str(e)}\n Please try rephrasing your question or check if your data is properly loaded."

    def clear_memory(self):
        """
        Clear the conversation memory
        """
        self.memory.clear()

    def get_conversation_history(self) -> List[BaseMessage]:
        """
        Get the current conversation history
        """
        return self.memory.chat_memory.messages