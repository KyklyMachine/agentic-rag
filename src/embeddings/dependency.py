from typing import Annotated

from fastapi import Depends, Request

from .repository import Embedder


def get_embedder(request: Request) -> Embedder:
    return request.app.state.embedder

EmbedderDep = Annotated[Embedder, Depends(get_embedder)]
