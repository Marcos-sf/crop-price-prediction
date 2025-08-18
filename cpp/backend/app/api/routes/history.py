# import os
# import pandas as pd
# from fastapi import Query

# @router.get("/history")
# async def get_history(
#     crop: str = Query(...),
#     mandi: str = Query(...)
# ):
#     """
#     Return real historical mandi price data for given crop and mandi.
#     """
#     try:
#         # Build file path (adjust according to your storage)
#         file_path = f"data/{crop.lower()}_{mandi.lower()}.csv"

#         if not os.path.exists(file_path):
#             return {"history": []}  # No data

#         # Load CSV
#         df = pd.read_csv(file_path)

#         # Ensure columns exist
#         if 'date' not in df.columns or 'modal_price' not in df.columns:
#             return {"history": []}

#         # Convert to dict
#         history = df[['date', 'modal_price']].to_dict(orient="records")

#         return {"history": history}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to load history: {str(e)}")
