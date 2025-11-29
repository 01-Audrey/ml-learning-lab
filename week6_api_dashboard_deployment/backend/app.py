"""
FastAPI Backend for AI Security System
Docker-optimized version
"""

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path

# Import from Day 37 (simplified for Docker)
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
import secrets

# JWT
from jose import JWTError, jwt

# FastAPI app
app = FastAPI(
    title="AI Security System API",
    description="Dockerized REST API",
    version="2.0.0"
)

# Database setup
DATABASE_URL = "sqlite:///./volumes/database/security_system.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Auth config
SECRET_KEY = secrets.token_urlsafe(32)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Password hashing (SHA256)
def get_password_hash(password: str) -> str:
    salt = "security_system_salt_2025"
    return hashlib.sha256((password + salt).encode()).hexdigest()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return get_password_hash(plain_password) == hashed_password

# JWT token
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# Database Models (from Day 37)
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class Person(Base):
    __tablename__ = "persons"
    id = Column(Integer, primary_key=True, index=True)
    person_id = Column(String(100), unique=True, index=True, nullable=False)
    name = Column(String(100), nullable=False)
    face_count = Column(Integer, default=0)
    added_date = Column(DateTime, default=datetime.utcnow)
    person_metadata = Column(Text)
    embeddings = relationship("FaceEmbedding", back_populates="person", cascade="all, delete-orphan")

class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"
    id = Column(Integer, primary_key=True, index=True)
    person_id = Column(String(100), ForeignKey("persons.person_id"), nullable=False)
    embedding = Column(Text, nullable=False)
    image_path = Column(String(255))
    quality_score = Column(Float)
    added_date = Column(DateTime, default=datetime.utcnow)
    person = relationship("Person", back_populates="embeddings")

class Alert(Base):
    __tablename__ = "alerts"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    alert_type = Column(String(50), nullable=False)
    priority = Column(String(20), nullable=False)
    person_id = Column(String(100))
    person_name = Column(String(100))
    location = Column(String(100))
    description = Column(Text)
    image_path = Column(String(255))
    acknowledged = Column(Boolean, default=False)
    acknowledged_by = Column(String(100))
    acknowledged_at = Column(DateTime)
    resolved = Column(Boolean, default=False)
    resolved_by = Column(String(100))
    resolved_at = Column(DateTime)
    resolved_notes = Column(Text)
    escalation_level = Column(Integer, default=1)
    escalation_timestamp = Column(DateTime)

# Create tables
Base.metadata.create_all(bind=engine)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic models
class Token(BaseModel):
    access_token: str
    token_type: str

class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    is_active: bool
    is_admin: bool
    created_at: datetime

    class Config:
        from_attributes = True

# Helper functions
def get_user_by_username(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()

def create_user(db: Session, username: str, email: str, password: str, is_admin: bool = False):
    hashed = get_password_hash(password)
    user = User(username=username, email=email, hashed_password=hashed, is_admin=is_admin)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

def authenticate_user(db: Session, username: str, password: str):
    user = get_user_by_username(db, username)
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid credentials")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    user = get_user_by_username(db, username)
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

# API Endpoints
@app.get("/")
def root():
    return {"status": "online", "message": "AI Security System API", "version": "2.0.0"}

@app.get("/api/v2/health")
def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database": "connected"
    }

@app.post("/api/v2/token", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/api/v2/register", response_model=UserResponse)
def register(user: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(
        (User.username == user.username) | (User.email == user.email)
    ).first()

    if existing:
        raise HTTPException(status_code=400, detail="User already exists")

    db_user = create_user(db, user.username, user.email, user.password)
    return db_user

@app.get("/api/v2/users/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    return current_user

@app.get("/api/v2/faces")
def get_faces(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    persons = db.query(Person).all()
    result = []
    for p in persons:
        result.append({
            "person_id": p.person_id,
            "name": p.name,
            "face_count": p.face_count,
            "added_date": p.added_date.isoformat(),
            "metadata": json.loads(p.person_metadata) if p.person_metadata else None
        })
    return {"status": "success", "total_persons": len(result), "persons": result}

@app.post("/api/v2/faces")
def add_face(
    person_id: str,
    name: str,
    metadata: Optional[dict] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    existing = db.query(Person).filter(Person.person_id == person_id).first()
    if existing:
        raise HTTPException(status_code=400, detail="Person already exists")

    person = Person(
        person_id=person_id,
        name=name,
        face_count=0,
        person_metadata=json.dumps(metadata) if metadata else None
    )
    db.add(person)
    db.commit()
    return {"status": "success", "person_id": person_id, "message": f"Person {name} added"}

@app.delete("/api/v2/faces/{person_id}")
def delete_face(person_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin only")

    person = db.query(Person).filter(Person.person_id == person_id).first()
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")

    db.delete(person)
    db.commit()
    return {"status": "success", "message": f"Person {person_id} deleted"}

@app.get("/api/v2/alerts")
def get_alerts(
    limit: int = 100,
    priority: Optional[str] = None,
    acknowledged: Optional[bool] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    query = db.query(Alert)
    if priority:
        query = query.filter(Alert.priority == priority)
    if acknowledged is not None:
        query = query.filter(Alert.acknowledged == acknowledged)

    alerts = query.order_by(Alert.timestamp.desc()).limit(limit).all()

    result = []
    for a in alerts:
        result.append({
            "alert_id": a.id,
            "timestamp": a.timestamp.isoformat(),
            "alert_type": a.alert_type,
            "priority": a.priority,
            "person_id": a.person_id,
            "person_name": a.person_name,
            "location": a.location,
            "description": a.description,
            "acknowledged": a.acknowledged,
            "acknowledged_by": a.acknowledged_by
        })

    return {"status": "success", "total_alerts": len(result), "alerts": result}

@app.post("/api/v2/alerts/acknowledge")
def ack_alert(
    alert_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    alert.acknowledged = True
    alert.acknowledged_by = current_user.username
    alert.acknowledged_at = datetime.utcnow()
    db.commit()

    return {"status": "success", "message": f"Alert {alert_id} acknowledged"}

# Startup: Create default admin user
@app.on_event("startup")
def startup():
    db = SessionLocal()
    try:
        admin = db.query(User).filter(User.username == "admin").first()
        if not admin:
            create_user(db, "admin", "admin@security.com", "pass123", is_admin=True)
            print("Default admin user created (admin/pass123)")

        alert_count = db.query(Alert).count()
        if alert_count == 0:
            alert = Alert(
                alert_type="unknown_person",
                priority="critical",
                person_id="unknown_1",
                person_name="Unknown Person",
                location="main_entrance",
                description="Unknown person detected"
            )
            db.add(alert)
            db.commit()
            print("Sample alert created")
    finally:
        db.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
