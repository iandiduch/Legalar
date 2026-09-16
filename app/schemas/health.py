from pydantic import BaseModel


class ServiceStatus(BaseModel):
    name: str
    reachable: bool
    detail: str | None = None


class HealthCheckResponse(BaseModel):
    status: str
    services: list[ServiceStatus]
