from fastapi import HTTPException


class IndexIsNotExist(Exception): ...

class ServiceUnavaliable(Exception): ...

service_unavaliable_http_exception = HTTPException(
    status_code=503,
    detail={"error": "infra_unavailable", "service": "Qdrant", "message": "Connection timeout"}
    )