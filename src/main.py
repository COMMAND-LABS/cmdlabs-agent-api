"""
Completion API - streaming LLM completion microservice.
Exposes the streaming agent endpoints, split out from the main AI API for
independent scaling.
"""
from dotenv import load_dotenv
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from src.logging_config import configure_logging
from src.middleware.dynamic_cors import DynamicCORSMiddleware
from src.ratelimit import limiter
from src.routers import healthcheck
from src.routers.agents.contact_chat import router as contact_chat_router
from src.routers.agents.router import router as agents_router

load_dotenv()
configure_logging()

app = FastAPI(
    title="Kalygo3 Completion API",
    description="Streaming agent completion microservice",
    docs_url="/api/docs",
    redoc_url=None,
    redirect_slashes=True,
)
app.root_path = ""

jwt_allowed_origins = [
    "https://kalygo.io",
    "http://127.0.0.1:3000",
    "http://localhost:3000",
    "https://kalygo-nextjs-service-830723611668.us-east1.run.app",
    "https://localhost:3000",
    "http://localhost:5000",
]

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(
    DynamicCORSMiddleware,
    allowed_origins=jwt_allowed_origins,
    allow_credentials=True,
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "message": "The request could not be processed due to validation errors.",
            "details": [{"location": " -> ".join(str(loc) for loc in e["loc"]), "message": e["msg"]} for e in errors],
            "path": str(request.url.path),
        },
    )

app.include_router(healthcheck.router, prefix="")
app.include_router(agents_router, prefix="/api/agents", tags=["Agents"])
app.include_router(contact_chat_router, prefix="/api/contact-chat", tags=["Contact Chat"])
