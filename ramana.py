import os
import google.generativeai as palm
from dotenv import load_dotenv
from pgvector.sqlalchemy import Vector
from sqlalchemy import create_engine, insert, select, text, Integer, String, Text
from sqlalchemy.orm import declarative_base, mapped_column, Session

load_dotenv()  # take environment variables from .env.
palm.configure(api_key=os.environ['API_KEY'])

models = [m for m in palm.list_models() if 'embedText' in m.supported_generation_methods]
model = models[0]

engine = create_engine('postgresql+psycopg2://postgres:password@localhost/postgres')
session = Session(engine)

Base = declarative_base()


class Document(Base):
    __tablename__ = 'document'

    id = mapped_column(Integer, primary_key=True)
    content = mapped_column(Text)
    embedding = mapped_column(Vector(768))


vectors = palm.generate_embeddings(model=model, text="where is andra")
doc = Document(embedding=vectors["embedding"])
neighbors = session.scalars(select(Document).order_by(Document.embedding.max_inner_product(doc.embedding)).limit(5))
for neighbor in neighbors:
    print(neighbor.content)
