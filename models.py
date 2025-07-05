
# tablas y helpers de base de datos (SQLModel)

from sqlmodel import SQLModel, Field, Relationship, Session, create_engine
from typing   import Optional, List
from datetime import datetime
import os

# -----------------------------------
# ---------- Configuración ----------
# -----------------------------------
DB_URL = os.getenv("DB_URL", "sqlite:///users.db")
engine = create_engine(DB_URL, echo=False)


# --------------------------------------
# ---------- Modelos de datos ----------
# --------------------------------------
class User(SQLModel, table=True):
    id          : Optional[int] = Field(default=None, primary_key=True)
    email       : str           = Field(unique=True, index=True)
    hashed      : str
    is_premium  : bool          = Field(default=False)
    created_at  : datetime      = Field(default_factory=datetime.utcnow)

    history  : List["OcrRecord"] = Relationship(back_populates="user")
    payments : List["Payment"]   = Relationship(back_populates="user")


class OcrRecord(SQLModel, table=True):
    id         : Optional[int] = Field(default=None, primary_key=True)
    text       : str
    created_at : datetime      = Field(default_factory=datetime.utcnow)

    user_id : int    = Field(foreign_key="user.id")
    user    : "User" = Relationship(back_populates="history")


class Payment(SQLModel, table=True):
    id           : Optional[int] = Field(default=None, primary_key=True)
    provider_id  : str           = Field(index=True)
    user_id      : int           = Field(foreign_key="user.id", index=True)
    user         : "User"        = Relationship(back_populates="payments")

    amount       : int           # céntimos
    currency     : str           = Field(default="eur")
    status       : str
    payment_method : Optional[str] = None

    created_at   : datetime      = Field(default_factory=datetime.utcnow)
    updated_at   : datetime      = Field(default_factory=datetime.utcnow)


# ------------------------------------------------
# ---------- Funciones de base de datos ----------
# ------------------------------------------------
def create_db_and_tables() -> None:
    SQLModel.metadata.create_all(engine)


def get_session():
    """Dependency FastAPI – yields an opened Session"""
    with Session(engine) as session:
        yield session
