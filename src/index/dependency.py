from typing import Annotated

from fastapi import Depends, Request

from .repository import VectorDBRepository


def get_vector_db(request: Request) -> VectorDBRepository:
    return request.app.state.vector_db

VectorDBDep = Annotated[VectorDBRepository, Depends(get_vector_db)]
