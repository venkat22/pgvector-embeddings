import os
import google.generativeai as palm
from dotenv import load_dotenv
from pgvector.sqlalchemy import Vector
from sqlalchemy import create_engine, insert, select, text, Integer, String, Text
from sqlalchemy.orm import declarative_base, mapped_column, Session
import pandas as pd
import pprint

load_dotenv()  # take environment variables from .env.

palm.configure(api_key=os.environ['API_KEY'])

response = palm.generate_text(prompt="The opposite of hot is")
print(response.result)  # 'cold.'



for model in palm.list_models():
    pprint.pprint(model)  # 🦎🦦🦬🦄

models = [m for m in palm.list_models() if 'embedText' in m.supported_generation_methods]
model = models[0]

engine = create_engine('postgresql+psycopg2://postgres:password@localhost/postgres')
with engine.connect() as conn:
    conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
    conn.commit()
Base = declarative_base()


class Document(Base):
    __tablename__ = 'document'

    id = mapped_column(Integer, primary_key=True)
    content = mapped_column(Text)
    embedding = mapped_column(Vector(768))


Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)

pd.set_option('display.max_colwidth', 10)
df = pd.read_csv("questions.csv", usecols=["qid1", "question1"], index_col=False)
df = df.sample(frac=1).reset_index(drop=True)
df_questions_imp = df[:199]

session = Session(engine)

for index, row in df_questions_imp.iterrows():
    vectors = palm.generate_embeddings(model=model, text=row["question1"])
    session.add(Document(content=row["question1"], embedding=vectors["embedding"]));
    print(index)
session.commit()