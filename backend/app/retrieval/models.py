from pydantic import BaseModel


class SearchRequest(BaseModel):
    query: str
    limit: int = 5
    source_filename: str | None = None
