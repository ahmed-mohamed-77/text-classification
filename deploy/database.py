from sqlalchemy import create_engine, Engine, Column, INTEGER, VARCHAR, Numeric
from sqlalchemy.orm import declarative_base
from dotenv import load_dotenv
import os

load_dotenv()

URL = os.getenv("POSTGRES_URL")

engine: Engine = create_engine(url=URL, echo=True)

Base = declarative_base()

class ClassifierResponse(Base):
    __tablename__ = "qa"
    id = Column(INTEGER, primary_key=True, autoincrement=True)
    question = Column(VARCHAR(100), nullable=False)
    label = Column(VARCHAR(10), nullable=False)
    score = Column(Numeric(10, 9), nullable=False)