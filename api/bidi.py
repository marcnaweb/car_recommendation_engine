from fastapi import FastAPI, HTTPException
from prepearing_data.recommendation import show_similar_cars  # Ensure this import is correct based on your file's structure
from pydantic import BaseModel

app = FastAPI()

# Define the path operation function
@app.get("/recommendations/{car_code}")
async def get_recommendations(car_code: str):  # Path parameters are received as strings
    try:
        # Convert car_code to integer since the expected type is int64
        car_code_int = int(car_code)
        recommendations = show_similar_cars(car_code_int)
        # Assuming recommendations is a DataFrame and you want to return JSON
        return recommendations.to_dict(orient='records')
    except ValueError:
        # If the conversion fails, it means the car_code was not a valid integer
        raise HTTPException(status_code=400, detail="car_code must be an integer")
    except Exception as e:
        # Handle other exceptions, e.g., if show_similar_cars raises an error
        raise HTTPException(status_code=404, detail=str(e))
