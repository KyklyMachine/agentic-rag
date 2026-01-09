from fastapi import HTTPException


class IndexIsNotExist(Exception): ...

class ServiceUnavaliable(Exception): ...

class InconsistentIndex(Exception): ...

class IndexNotFoundException(Exception): ...

service_unavaliable_http_exception = HTTPException(
    status_code=503,
    detail={"error": "infra_unavailable", "message": "Connection timeout"}
    )
