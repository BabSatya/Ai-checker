from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from firebase_admin import auth
from backend.firebase_admin import db

admin_router = APIRouter(
    prefix="/admin",
    tags=["Admin Panel"]
)


# ============================
# Request Schema
# ============================
class CreateUserRequest(BaseModel):
    email: str
    password: str
    tokens: int


# ============================
# Create User API
# ============================
@admin_router.post("/create-user")
def create_user(data: CreateUserRequest):

    try:
        # 1. Create Firebase Auth User
        user = auth.create_user(
            email=data.email,
            password=data.password
        )

        # 2. Store Token Info in Firestore
        db.collection("users").document(user.uid).set({
            "email": data.email,
            "tokens": data.tokens,
            "role": "user"
        })

        return {
            "message": "✅ User Created Successfully",
            "uid": user.uid,
            "tokens": data.tokens
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
