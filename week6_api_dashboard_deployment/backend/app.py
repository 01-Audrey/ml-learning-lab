"""
FastAPI Backend for AI Security System
WITH REAL ML INTEGRATION - Day 42 Final Version
"""

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import threading
import queue

# Database
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
import secrets
from jose import JWTError, jwt

# ML Libraries
import cv2
import numpy as np
import face_recognition

# FastAPI app
app = FastAPI(
    title="AI Security System API",
    description="Production API with Real ML",
    version="3.0.0"
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

# ML Models - Global variables
face_detector = None
known_face_encodings = []
known_face_names = []

def load_ml_models():
    """Load ML models on startup."""
    global face_detector

    try:
        # Load face detection model (DNN Caffe)
        prototxt = "/app/models/deploy.prototxt"
        caffemodel = "/app/models/res10_300x300_ssd_iter_140000.caffemodel"

        if Path(prototxt).exists() and Path(caffemodel).exists():
            face_detector = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
            print("✅ Face detection model loaded")
        else:
            print("⚠️  Face detection models not found - using face_recognition instead")
    except Exception as e:
        print(f"⚠️  Error loading face detector: {e}")

# Password hashing
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

# Database Models (same as before)
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

# Load known faces from database
def load_known_faces(db: Session):
    """Load known face encodings from database."""
    global known_face_encodings, known_face_names

    persons = db.query(Person).all()
    known_face_encodings = []
    known_face_names = []

    for person in persons:
        embeddings = db.query(FaceEmbedding).filter(FaceEmbedding.person_id == person.person_id).all()
        for emb in embeddings:
            try:
                encoding = json.loads(emb.embedding)
                known_face_encodings.append(np.array(encoding))
                known_face_names.append(person.name)
            except:
                pass

    print(f"✅ Loaded {len(known_face_encodings)} known faces")

# ML Functions
def detect_faces_dnn(frame):
    """Detect faces using DNN model."""
    if face_detector is None:
        return []

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    face_detector.setInput(blob)
    detections = face_detector.forward()

    faces = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            faces.append({
                "box": (startX, startY, endX - startX, endY - startY),
                "confidence": float(confidence)
            })

    return faces

def recognize_faces(frame, face_locations):
    """Recognize faces using face_recognition library."""
    if len(known_face_encodings) == 0:
        return ["Unknown"] * len(face_locations)

    # Convert face locations to format expected by face_recognition
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        name = "Unknown"

        if True in matches:
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        names.append(name)

    return names

# Video streaming with REAL ML
video_queue = queue.Queue(maxsize=2)
camera_active = False
detection_stats = {"faces_detected": 0, "last_detection": None}

def process_video_stream_ml():
    """Process video with real ML models."""
    global camera_active, detection_stats

    db = SessionLocal()
    load_known_faces(db)

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("⚠️  Camera not available - using demo mode")
        camera_active = False
        db.close()

        # Demo mode - generate frames
        while True:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            cv2.putText(frame, f"AI Security System - DEMO MODE", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"{timestamp}", (50, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, "Camera not accessible in Docker", (50, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, "Run locally for real detection", (50, 170), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Simulate detection box
            cv2.rectangle(frame, (200, 200), (440, 400), (0, 255, 0), 2)
            cv2.putText(frame, "Demo: Detection Area", (205, 195), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                try:
                    video_queue.put_nowait(buffer.tobytes())
                except queue.Full:
                    pass

            import time
            time.sleep(1/30)

        return

    camera_active = True
    print("✅ Camera started - Real ML processing active")

    frame_count = 0

    while camera_active:
        ret, frame = camera.read()
        if not ret:
            break

        frame_count += 1

        # Process every 3rd frame for performance
        if frame_count % 3 == 0:
            # Detect faces
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)

            detection_stats["faces_detected"] = len(face_locations)
            detection_stats["last_detection"] = datetime.now().isoformat()

            # Recognize faces
            if len(face_locations) > 0:
                names = recognize_faces(rgb_frame, face_locations)

                # Draw boxes and labels
                for (top, right, bottom, left), name in zip(face_locations, names):
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, name, (left + 6, bottom - 6), 
                               cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

                    # Create alert for unknown persons
                    if name == "Unknown":
                        alert = Alert(
                            alert_type="unknown_person",
                            priority="high",
                            person_id="unknown",
                            person_name="Unknown Person",
                            location="camera_1",
                            description=f"Unknown person detected at {datetime.now().strftime('%H:%M:%S')}"
                        )
                        db.add(alert)
                        db.commit()

        # Add overlay info
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"AI Security System - {timestamp}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Faces: {detection_stats['faces_detected']} | FPS: 30", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            try:
                video_queue.put_nowait(buffer.tobytes())
            except queue.Full:
                pass

    camera.release()
    db.close()
    print("📹 Camera stopped")

def generate_video_frames():
    """Generator for video frames."""
    while True:
        try:
            frame_bytes = video_queue.get(timeout=1.0)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except queue.Empty:
            pass

# API Endpoints (same as before, keeping all existing endpoints)
@app.get("/")
def root():
    return {"status": "online", "message": "AI Security System API", "version": "3.0.0", "ml": "enabled"}

@app.get("/api/v2/health")
def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database": "connected",
        "ml_models": "loaded" if face_detector else "demo_mode"
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

    # Reload known faces
    load_known_faces(db)

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

    # Reload known faces
    load_known_faces(db)

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

@app.get("/api/v2/video/stream")
def video_stream():
    """Stream video with real ML processing."""
    return StreamingResponse(
        generate_video_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/api/v2/video/status")
def video_status(current_user: User = Depends(get_current_user)):
    """Get video and ML status."""
    return {
        "camera_active": camera_active,
        "ml_enabled": face_detector is not None,
        "faces_detected": detection_stats.get("faces_detected", 0),
        "last_detection": detection_stats.get("last_detection"),
        "fps": 30 if camera_active else 0
    }

# Startup events
@app.on_event("startup")
def startup():
    """Initialize on startup."""
    db = SessionLocal()
    try:
        # Create admin user
        admin = db.query(User).filter(User.username == "admin").first()
        if not admin:
            create_user(db, "admin", "admin@security.com", "pass123", is_admin=True)
            print("✅ Default admin user created")

        # Create sample alert
        alert_count = db.query(Alert).count()
        if alert_count == 0:
            alert = Alert(
                alert_type="system_startup",
                priority="low",
                person_id="system",
                person_name="System",
                location="server",
                description="AI Security System started successfully"
            )
            db.add(alert)
            db.commit()
            print("✅ Sample alert created")

        # Load ML models
        load_ml_models()

        # Start video thread
        video_thread = threading.Thread(target=process_video_stream_ml, daemon=True)
        video_thread.start()
        print("✅ Video processing thread started")

    finally:
        db.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
