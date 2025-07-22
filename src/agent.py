#%%
import os
from langchain import hub
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from openai import OpenAI
from dotenv import load_dotenv

from src.crud import CRUD
from src.eda import EDAFeatures
from utils import stderr_print
#%%

class ChatBot:
    def __init__(self, past_know: str, pred_know: str):
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.past_know: str = past_know
        self.pred_know: str = pred_know

    def set_past_know(self, past_know: str):
        self.past_know = past_know

    def set_pred_know(self, pred_know: str):
        self.pred_know = pred_know

    def get_gpt35_response(self, question: str, temperature: float = 0.7, max_tokens: int = 1024, model: str = "gpt-3.5-turbo") -> str:
        past_know = self.past_know
        pred_know = self.pred_know
        client = self.client
        stderr_print(f"Past know: {past_know}\nPred know: {pred_know}\nQuestion: {question}")
        prompt = f"""
You are 'RetailMind,' an expert AI business consultant. Your goal is to provide concise, data-driven analysis and strategic advice to retail decision-makers.

You have been provided with two key data summaries:
- **Historical Sales Data:**
{past_know}
- **Predicted Sales Forecast:**
{pred_know}

Based ONLY on the information in these summaries, you must synthesize the data, identify key business insights, and answer the user's question. If the data is insufficient to answer, state that clearly.

Structure your response using the following professional format:

Executive Summary
[Directly and concisely answer the user's question here.]

---

Key Analytical Insights
- **Performance Trend:** Compare a key metric (e.g., average sales, peak/low points) from the historical data against the forecast. Is performance expected to improve, decline, or stay consistent?
- **Holiday & Anomaly Impact:** State the observed or predicted impact of holidays on sales. Mention any standout peaks or dips.
- **Correlation Insights:** Highlight the most significant relationship noted in the data (e.g., with unemployment) and identify any stores that are strong outliers.

---

Strategic Recommendations
- Based on your analysis, provide one or two concrete, actionable recommendations. This could be a potential business opportunity to explore or a risk to investigate further.

Maintain a professional and direct tone. Use the specific figures from the provided summaries to support your analysis.

**User Question:** {question}
"""

        response = client.chat.completions.create(model=model,
                                                  messages=[
                                                      {"role": "user", "content": prompt}
                                                  ],
                                                  temperature=temperature,
                                                  max_tokens=max_tokens)
        return response.choices[0].message.content.strip()

#%%

class RetailAgent:
    """
    AI agent that uses CRUD and EDA to answer questions about the retail dataset.
    """
    def __init__(self, crud_obj: CRUD, eda_obj: EDAFeatures):
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
        ]

        prompt = hub.pull("hwchase17/openai-tools-agent")
        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        self.executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True)

    def ask(self, question: str) -> str:
        """
        Asks the agent a question and returns an answer.
        :param question (str): The user's question in natural language.
        :return: answer
        """
        response = self.executor.invoke({"input": question})
        return response['output']

#%%
print(os.getcwd())
crud_manager = CRUD()
eda_analyzer = EDAFeatures(gen_df=crud_manager.gen_df, spec_df=crud_manager.spec_df)
#%%
retail_chatbot = RetailAgent(crud_obj=crud_manager, eda_obj=eda_analyzer)
#%%

question1 = "What were the total sales for store 2 in 2011?"
answer1 = retail_chatbot.ask(question1)
print(f"\nAnswer: {answer1}")
#%%
question2 = "Show me an analysis of the top 3 performing stores."
answer2 = retail_chatbot.ask(question2)
print(f"\nAnswer: {answer2}")
#%%
question3 = "Tell me everything you can about this data, including stores, holiday performance, top performers, retail headwinds, etc."
answer3 = retail_chatbot.ask(question3)
print(f"\nAnswer: {answer3}")