# FastAPI + JWT + Stripe
import os, json, logging, time
from datetime           import datetime, timedelta
from typing             import List, Annotated

import stripe
from jose               import jwt, JWTError
from passlib.context    import CryptContext
from fastapi            import (FastAPI, Depends, UploadFile, File,
                                HTTPException, status, Query)
from fastapi.security   import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel           import select, delete, Session
from pydantic           import BaseModel

from google.cloud       import vision
from google.cloud.vision import AnnotateImageRequest, Feature
from google.api_core    import exceptions as gv_exceptions

from models             import (create_db_and_tables, get_session,
                                User, OcrRecord, Payment)
import ocr_engine as engine   # ← motor local
 
#-----------------------------
#----Configuración general----
#-----------------------------


SECRET_KEY  = os.getenv("SECRET_KEY", "1234567890abcdef")
ALGORITHM   = "HS256"
ACCESS_TTL  = 60 * 24          # minutos

stripe.api_key = os.getenv("STRIPE_SECRET_KEY", "")
pwd_ctx   = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2    = OAuth2PasswordBearer(tokenUrl="/login")

# ---------------------------
# -------Google Vision-------
# ---------------------------
try:
    _gv_client = vision.ImageAnnotatorClient()
except gv_exceptions.DefaultCredentialsError:
    _gv_client = None
LANGS = os.getenv("EASYOCR_LANGS", "es,en").split(",")

# ----------------------------
# ---Configuración FastAPI----
# ----------------------------

app = FastAPI(title="OCR API – 3 capas")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

create_db_and_tables()
logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s %(message)s")
# --------------------------------
# ----Helpers de autenticación----
# --------------------------------
def hash_pw(pw):            return pwd_ctx.hash(pw)
def verify_pw(pw, h):       return pwd_ctx.verify(pw, h)

def create_token(user: User) -> str:
    exp = datetime.utcnow() + timedelta(minutes=ACCESS_TTL)
    return jwt.encode({"sub": user.email,
                       "is_premium": user.is_premium,
                       "exp": exp}, SECRET_KEY, ALGORITHM)

def get_current_user(token: str = Depends(oauth2),
                     sess : Session = Depends(get_session)) -> User:
    exc = HTTPException(status.HTTP_401_UNAUTHORIZED, "Credenciales no válidas",
                        headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email   = payload.get("sub")
    except JWTError:
        raise exc
    user = sess.exec(select(User).where(User.email == email)).first()
    if not user:
        raise exc
    return user

# -----------------------------------
# ----Modelos de entrada y salida----
# -----------------------------------
class RegisterIn(BaseModel): email: str; password: str
class LoginIn(RegisterIn):   pass
class UpgradeIn(BaseModel):  payment_intent_id: str

# #####################################
# ########## Rutas de la API ##########
# #####################################

# -----------------------------------
# ----Endpoints de autenticación-----
# -----------------------------------

@app.post("/register", status_code=201, tags=["auth"])
def register(data: RegisterIn, sess=Depends(get_session)):
    if sess.exec(select(User).where(User.email == data.email)).first():
        raise HTTPException(409, "Email ya registrado")
    sess.add(User(email=data.email, hashed=hash_pw(data.password)))
    sess.commit()
    return {"msg": "Registro ok"}

@app.post("/login", tags=["auth"])
def login(data: LoginIn, sess=Depends(get_session)):
    user = sess.exec(select(User).where(User.email == data.email)).first()
    if not user or not verify_pw(data.password, user.hashed):
        raise HTTPException(400, "Credenciales incorrectas")
    return {"token": create_token(user)}

# -----------------------------------
# ----Endpoints de OCR (local)-------
# -----------------------------------
@app.post("/ocr", tags=["ocr"])
async def ocr_endpoint(file: UploadFile = File(...),
                       user = Depends(get_current_user),
                       sess = Depends(get_session)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(415, "El archivo debe ser imagen")
    try:
        text, latency = engine.recognise(await file.read())
    except Exception as e:
        raise HTTPException(500, f"OCR error: {e}")
    sess.add(OcrRecord(text=text, user_id=user.id)); sess.commit()
    return {"text": text, "latency_ms": latency, "user": user.email}

@app.post("/ocr_batch", tags=["ocr"])
async def ocr_batch(files: List[UploadFile] = File(...),
                    user  = Depends(get_current_user),
                    sess  = Depends(get_session)):
    if not files:
        raise HTTPException(400, "Sube al menos una imagen")
    imgs   = [await f.read() for f in files]
    texts, elapsed = engine.recognise_batch(imgs)
    for t in texts:
        sess.add(OcrRecord(text=t, user_id=user.id))
    sess.commit()
    return {"n": len(texts), "elapsed_ms": elapsed,
            "texts": texts, "user": user.email}

# -----------------------------------
# --Endpoints de OCR (Google Vision)-
# -----------------------------------
def _google_vision_ocr(jpg_bytes: bytes) -> str:
    if _gv_client is None:
        raise RuntimeError("Google Vision no configurado")
    req = AnnotateImageRequest(
        image     = vision.Image(content=jpg_bytes),
        features  = [Feature(type_=Feature.Type.DOCUMENT_TEXT_DETECTION)],
        image_context = {"language_hints": LANGS}
    )
    resp = _gv_client.annotate_image(request=req)
    if resp.error.message:
        raise RuntimeError(resp.error.message)
    return resp.full_text_annotation.text or ""

@app.post("/ocr_google", tags=["ocr"])
async def ocr_google(file: UploadFile = File(...),
                     user = Depends(get_current_user),
                     sess = Depends(get_session)):
    if not user.is_premium:
        raise HTTPException(403, "Solo usuarios premium")
    if not file.content_type.startswith("image/"):
        raise HTTPException(415, "Debe subir imagen")
    img_bytes = await file.read()

    # recortamos con el mismo motor
    rgb      = engine.bytes_to_rgb(img_bytes)
    cropped  = engine._crop_table(rgb)        # pylint: disable=protected-access
    _, buf   = engine.cv2.imencode(".jpg", engine.cv2.cvtColor(
                                        cropped, engine.cv2.COLOR_RGB2BGR))
    t0       = time.perf_counter()
    try:
        text = _google_vision_ocr(buf.tobytes())
    except Exception as e:
        raise HTTPException(500, f"Vision error: {e}")
    lat = (time.perf_counter() - t0)*1000
    sess.add(OcrRecord(text=text, user_id=user.id)); sess.commit()
    return {"text": text, "latency_ms": round(lat,2), "user": user.email}

@app.post("/ocr_batch_google", tags=["ocr"])
async def ocr_batch_google(
    files: List[UploadFile] = File(...),
    user  : User           = Depends(get_current_user),
    sess  : Session        = Depends(get_session)
):
    # 1) validacion de entrada
    if _gv_client is None:
        raise HTTPException(500, "Google Vision no está disponible")
    if not user.is_premium:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Solo usuarios premium pueden usar Google Vision"
        )
    if not files:
        raise HTTPException(400, "Debes subir al menos una imagen")

    # 2) construir la lista de peticiones para Vision
    requests: List[AnnotateImageRequest] = []
    for f in files:
        img_bytes = await f.read()
        try:
            # convertimos bytes a RGB
            rgb = engine.bytes_to_rgb(img_bytes)
        except Exception as e:
            raise HTTPException(400, f"Imagen no válida: {e}")

        cropped = engine._crop_table(rgb)  # pylint: disable=protected-access
        _, buf  = engine.cv2.imencode(
            ".jpg", engine.cv2.cvtColor(cropped, engine.cv2.COLOR_RGB2BGR))

        requests.append(AnnotateImageRequest(
            image = vision.Image(content=buf.tobytes()),
            features=[Feature(type_=Feature.Type.DOCUMENT_TEXT_DETECTION)],
            image_context={"language_hints": LANGS}
        ))

    # 3) llamada batch a Vision
    t0 = time.perf_counter()
    try:
        responses = _gv_client.batch_annotate_images(requests=requests).responses
    except Exception as e:
        raise HTTPException(500, f"Google Vision batch error: {e}")

    # 4) procesar respuestas y guardar en BBDD
    texts: List[str] = []
    for resp in responses:
        if resp.error.message:
            raise HTTPException(500, f"Vision error: {resp.error.message}")
        txt = resp.full_text_annotation.text or ""
        texts.append(txt)
        sess.add(OcrRecord(text=txt, user_id=user.id))
    sess.commit()

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    return {
        "n"          : len(texts),
        "elapsed_ms" : elapsed_ms,
        "texts"      : texts,
        "user"       : user.email
    }

# -----------------------------------
# ----Historial de OCR (local)-------
# -----------------------------------
@app.get("/history", tags=["ocr"])
def history(limit: Annotated[int, Query(le=100)] = 20,
            user = Depends(get_current_user),
            sess = Depends(get_session)):
    q = (select(OcrRecord)
         .where(OcrRecord.user_id == user.id)
         .order_by(OcrRecord.created_at.desc())
         .limit(limit))
    rows = sess.exec(q).all()
    return [
        {
            "id"  : r.id,
            "text": r.text,
            "date": r.created_at.isoformat(timespec="seconds")
        } for r in rows
    ]

@app.delete("/history/{rec_id}", status_code=204, tags=["ocr"])
def delete_rec(rec_id: int, user=Depends(get_current_user),
               sess=Depends(get_session)):
    rec = sess.get(OcrRecord, rec_id)
    if not rec or rec.user_id != user.id:
        raise HTTPException(404, "No encontrado")
    sess.delete(rec); sess.commit()

@app.delete("/history", status_code=204, tags=["ocr"])
def clear_history(user=Depends(get_current_user),
                  sess=Depends(get_session)):
    sess.exec(delete(OcrRecord).where(OcrRecord.user_id == user.id))
    sess.commit()

# -----------------------------------
# ----Endpoints de pagos (Stripe)----
# -----------------------------------
@app.post("/create-payment-intent", tags=["payments"])
def create_pi(user=Depends(get_current_user)):
    try:
        intent = stripe.PaymentIntent.create(
            amount=999, currency="eur", payment_method_types=["card"],
            metadata={"user_id": user.id})
    except stripe.error.StripeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.user_message or str(e)
        )
    return {"client_secret": intent.client_secret}


@app.post("/upgrade", tags=["payments"])
def upgrade(data: UpgradeIn, user=Depends(get_current_user),
            sess=Depends(get_session)):
    intent = stripe.PaymentIntent.retrieve(data.payment_intent_id)
    if intent.status != "succeeded":
        raise HTTPException(400, "Pago no completado")
    user.is_premium = True
    sess.add_all([user, Payment(provider_id=intent.id, user_id=user.id,
                                amount=intent.amount, currency=intent.currency,
                                status=intent.status,
                                payment_method=intent.payment_method)])
    sess.commit()
    return {"token": create_token(user)}

# -----------------------------------
# ----Health check y configuración----
# -----------------------------------
@app.get("/", tags=["health"])
def health():
    return {"status": "ok",
            "gpu": engine.GPU_FLAG,
            "langs": ",".join(engine.LANGS)}
