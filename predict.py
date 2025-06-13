# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input
from typing import List
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "Snowflake/snowflake-arctic-embed-m-v2.0"
MODEL_CACHE = "checkpoint"

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=MODEL_CACHE)
        self.model = AutoModel.from_pretrained(MODEL_ID, add_pooling_layer=False, cache_dir=MODEL_CACHE)

    def predict(
        self,
        documents: List[str] = Input(description="Document to create the embeddings for", default=['Snowflake is the Data Cloud!']),
    ) -> List[float]:
        encoded_input = self.tokenizer(documents, padding=True, return_tensors='pt')
        outputs = self.model(**encoded_input).last_hidden_state
        embeddings = outputs[:, 0].tolist()
        return embeddings
