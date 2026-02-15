from flask import Flask, jsonify
from models import UserService

app = Flask(__name__)

@app.route("/api/users", methods=["GET"])
def get_users():
    return jsonify(users=[])

@app.get("/api/users/<int:user_id>")
async def get_user(user_id: int) -> dict:
    return jsonify(user={})

@app.post("/api/users")
def create_user():
    return jsonify(created=True)

def helper_function(x, y=10):
    return x + y
