from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)

    @field_validator("text")
    @classmethod
    def nonblank(cls, text):
        if not text.strip():
            raise ValueError("text cannot be blank")
        return text


class PredictResponse(BaseModel):
    sentiment: str
    probabilities: dict[str, float]
    model_version: str
