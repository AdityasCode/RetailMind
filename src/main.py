from crud import parse_sales_csv
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from eda import generate_graphs
from ml import get_gpt35_response
import os
from fastapi import FastAPI
from pydantic import BaseModel

from utils import stderr_print

app = FastAPI()

past_insights: str = ""
pred_insights: str = ""

class QuestionRequest(BaseModel):
    question: str

@app.on_event("startup")
async def startup_event():
    global past_insights, pred_insights

    stderr_print("Loading data and models...")

    gen_df, spec_df = parse_sales_csv("../test_data/train.csv")
    store_id = 1
    os.environ['DYLD_LIBRARY_PATH'] = '/usr/local/Cellar/libomp/'
    store_df = gen_df[gen_df['Store'] == store_id][['Date', 'Weekly_Sales']].copy()
    train_data = TimeSeriesDataFrame.from_data_frame(
        gen_df,
        id_column="Store",
        timestamp_column="Date"
    )

    predictor = TimeSeriesPredictor.load("../autogluon-m4-hourly")
    predictions = predictor.predict(train_data)
    pred_df = predictions.reset_index()
    pred_df.rename(columns={"item_id":"Store", "timestamp":"Date", "0.9":"Weekly_Sales"}, inplace=True)
    test_data = train_data

    predictor.plot(test_data, predictions, quantile_levels=[0.1, 0.9], max_history_length=200, max_num_item_ids=4)

    # Generate insights
    past_insights = generate_graphs(gen_df, spec_df, isPred=0)
    pred_insights = generate_graphs(pred_df, spec_df, isPred=1)

    print("Initialization complete!")

@app.get("/")
async def root():
    return {"message": "Sales Analysis API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "initialized": past_insights is not None}

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """
    Send a question and get the insights
    :param request: question in model
    :return: Response
    """
    if (not past_insights) or (not pred_insights):
        return {"error": "System not initialized yet."}

    try:
        answer = get_gpt35_response(
            past_know=past_insights,
            pred_know=pred_insights,
            question=request.question
        )
        return {"question": request.question, "answer": answer}
    except Exception as e:
        return {"error": f"Error processing query: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)