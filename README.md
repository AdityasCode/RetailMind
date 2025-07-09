# RetailMind
RetailMind: Self-Driven Analytics Engine for a Leading Midsize US Retailer

To run locally:
Clone, then type ```uvicorn src.main:app --reload``` into a Terminal from the root directory of the project.
Run testing/send_requests, or in another terminal:
```
curl -X POST "http://0.0.0.0:8080/ask" \
-H "Content-Type: application/json" \
-d '{"question": "What are the sales projections?"}'
```
