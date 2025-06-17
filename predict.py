# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, BaseModel
from typing import List
from sentence_transformers import SentenceTransformer

MODEL_ID = "Snowflake/snowflake-arctic-embed-m-v2.0"
MODEL_CACHE = "checkpoint"

class Output(BaseModel):
    embedding: List[float]

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = SentenceTransformer(MODEL_ID, trust_remote_code=True, cache_folder=MODEL_CACHE)

    def predict(
        self,
        document: str = Input(description="Documents to create the embeddings for", default='Snowflake is the Data Cloud!'),
        query: bool = Input(description="Whether to add query prefix", default=False),
    ) -> Output:
        if(query):
            embedding = self.model.encode(document, prompt_name="query")
        else:
            embedding = self.model.encode(document)
        return Output(embedding=embedding)
