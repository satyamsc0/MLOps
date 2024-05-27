from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.get("/")
async def say_hello():
    return {"result": "Welcome to FastAPI"}

@app.get("/hello")
async def say_hello():
    return {"result": "Hello from API"}

@app.get("/bye")
async def say_bye():
    return  {"result": "Goodbye from API"}