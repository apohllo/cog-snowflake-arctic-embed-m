# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input
from typing import List
from sentence_transformers import SentenceTransformer

MODEL_ID = "Snowflake/snowflake-arctic-embed-m-v2.0"
MODEL_CACHE = "checkpoint"

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = SentenceTransformer(MODEL_ID, trust_remote_code=True, cache_folder=MODEL_CACHE)

    def predict(
        self,
        documents: List[str] = Input(description="Document to create the embeddings for", default=['Snowflake is the Data Cloud!']),
        query: bool = Input(description="Whether to add query prefix"", default=False),
    ) -> List[float]:
        if(query):
            embeddings = self.model.encode(documents, prompt_name="query")
        else:
            embeddings = self.model.encode(documents)
        return embeddings
