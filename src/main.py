# from crud import CRUD
# from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
# from eda import EDAFeatures
# from agent import ChatBot
# import pandas as pd
# from fastapi import FastAPI
# from pydantic import BaseModel
#
# from utils import print_stderr
#
# app = FastAPI()
#
# past_insights: str = ""
# pred_insights: str = ""
#
# class QuestionRequest(BaseModel):
#     question: str
#
# @app.on_event("startup")
# async def startup_event():
#     global past_insights, pred_insights
#
#     print_stderr("Loading data and models...")
#
#     crudder = CRUD()
#     gen_df, spec_df = crudder.gen_df, crudder.spec_df
#     storeID = 1
#     # os.environ['DYLD_LIBRARY_PATH'] = '/usr/local/Cellar/libomp/'
#     store_df = gen_df[gen_df['Store'] == storeID][['Date', 'Weekly_Sales']].copy()
#     pred_df = await generate_pred_df(gen_df)
#
#     # Generate insights
#
#     featureGenerator = EDAFeatures(gen_df=gen_df, spec_df=spec_df)
#     featureGeneratorForecasted = EDAFeatures(gen_df=pred_df, spec_df=spec_df, isForecasted=1)
#     past_insights = featureGenerator.generate_graphs(storeID=[storeID])
#     pred_insights = featureGeneratorForecasted.generate_graphs()
#
#     print("Initialization complete!")
#
#
# async def generate_pred_df(gen_df: pd.DataFrame) -> pd.DataFrame:
#     train_data = TimeSeriesDataFrame.from_data_frame(
#         gen_df,
#         id_column="Store",
#         timestamp_column="Date"
#     )
#     predictor = TimeSeriesPredictor.load("autogluon-m4-hourly")
#     predictions = predictor.predict(train_data)
#     pred_df = predictions.reset_index()
#     # print(f"\n\n\n\n{pred_df}\n\n\n\n")
#     pred_df.rename(columns={"item_id": "Store", "timestamp": "Date", "0.9": "Weekly_Sales"}, inplace=True)
#     pred_df.drop(columns=["mean", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8"], inplace=True)
#     pred_df["Weekly_Sales"] = pred_df["Weekly_Sales"].apply(lambda x: round(x, 2))
#     test_data = train_data
#     predictor.plot(test_data, predictions, quantile_levels=[0.1, 0.9], max_history_length=200, max_num_item_ids=4)
#     return pred_df
#
#
# @app.get("/")
# async def root():
#     return {"message": "Sales Analysis API is running"}
#
# @app.get("/health")
# async def health_check():
#     return {"status": "healthy", "initialized": past_insights is not None}
#
# @app.post("/ask")
# async def ask_question(request: QuestionRequest):
#     """
#     Send a question and get the insights
#     :param request: question in model
#     :return: Response
#     """
#     if (not past_insights) or (not pred_insights):
#         return {"error": "System not initialized yet."}
#
#     try:
#         chatbot = ChatBot(past_know=past_insights, pred_know=pred_insights)
#         answer = chatbot.get_gpt35_response(
#             question=request.question
#         )
#         return {"question": request.question, "answer": answer}
#     except Exception as e:
#         return {"error": f"Error processing query: {str(e)}"}
#
# if __name__ == "__main__":
#     import uvicorn
#     print("Server is running.")
#     uvicorn.run(app, host="0.0.0.0", port=8080)
# # import os
# # print("Dir:")
# # print(os.getcwd())
# # os.chdir("..")
# # import crud
# # from src.utils import stderr_print
# #
# # crudder = crud.CRUD()
# # stderr_print(crudder.gen_df)
# # stderr_print(crudder.spec_df)