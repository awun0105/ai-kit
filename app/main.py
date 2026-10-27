"""This file contains the main application entry point."""

from contextlib import asynccontextmanager
from datetime import datetime

from dotenv import load_dotenv
from fastapi import (
    FastAPI,
    Request,
    status,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from asgi_correlation_id import CorrelationIdMiddleware

from app.api.v1.api import api_router
from app.api.v1.chatbot import agent
from app.core.cache import cache_service
from app.core.config import settings
from app.core.kit import (
    build_toolkit_from_settings,
    get_guardrails,
)
from app.core.limiter import limiter
from app.core.logging import logger
from app.core.metrics import setup_metrics
from app.core.middleware import (
    LoggingContextMiddleware,
    MetricsMiddleware,
    ProfilingMiddleware,
)
from app.core.observability import langfuse_init
from app.services.agents import ResilientChatModel
from app.services.database import database_service
from app.services.guardrails import guard_tools
from app.services.llm import llm_service
from app.services.memory import memory_service

# Load environment variables
load_dotenv()
langfuse_init()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown events."""
    logger.info(
        "application_startup",
        project_name=settings.PROJECT_NAME,
        version=settings.VERSION,
        api_prefix=settings.API_V1_STR,
    )

    # Initialize cache service (connects to Valkey if configured)
    try:
        await cache_service.initialize()
    except Exception as e:
        logger.exception("cache_initialization_failed", error=str(e))

    # Pre-warm the LangGraph agent: create graph + connection pool at startup
    # to avoid cold-start latency on the first request
    try:
        await agent.create_graph()
        logger.info("graph_pre_warmed")
    except Exception as e:
        logger.exception("graph_pre_warm_failed", error=str(e))

    # Pre-warm mem0 AsyncMemory: initializes pgvector connection and schema check
    # so the first search() cache miss or add() doesn't pay the ~130ms cold-init cost
    try:
        await memory_service.initialize()
    except Exception as e:
        logger.exception("memory_service_pre_warm_failed", error=str(e))

    # Let the model delegate to remote A2A agents (A2A_REMOTE_AGENTS).
    if settings.A2A_REMOTE_AGENTS:
        try:
            from app.services.a2a.client import (  # optional "a2a" extra
                build_remote_agent_tool,
                parse_remote_agents,
            )

            agent.register_tools([build_remote_agent_tool(parse_remote_agents(settings.A2A_REMOTE_AGENTS))])
        except Exception as e:
            logger.exception("a2a_remote_tool_failed", error=str(e))

    # Load tools from external MCP servers (MCP_CONFIG_FILE). A broken server must
    # not take the API down: the agent starts with its built-in tools instead.
    mcp_toolkit = None
    try:
        mcp_toolkit = build_toolkit_from_settings()
        if mcp_toolkit is not None:
            agent.register_tools(await mcp_toolkit.start())
    except Exception as e:
        logger.exception("mcp_toolkit_start_failed", error=str(e))

    # Give the agent a research team that can use every tool registered so far.
    if settings.DEEP_RESEARCH_ENABLED:
        try:
            from app.services.agents.research import build_deep_research_tool  # optional "deep" extra

            sources = [t for name, t in agent.tools_by_name.items() if name not in ("ask_human", "deep_research")]
            # Researchers call tools directly, outside the graph's tool node: wrap them in the same rails.
            agent.register_tools(
                [
                    build_deep_research_tool(
                        ResilientChatModel(service=llm_service),
                        guard_tools(sources, get_guardrails()),
                        settings.DEEP_RESEARCH_MAX_MODEL_CALLS,
                        settings.DEEP_RESEARCH_MAX_TOOL_CALLS,
                    )
                ]
            )
        except Exception as e:
            logger.exception("deep_research_tool_failed", error=str(e))

    yield

    # Cleanup on shutdown
    if mcp_toolkit is not None:
        await mcp_toolkit.stop()
    await cache_service.close()
    if agent._connection_pool:
        await agent._connection_pool.close()
        logger.info("connection_pool_closed")
    logger.info("application_shutdown")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

# Set up Prometheus metrics
setup_metrics(app)

# Add logging context middleware (must be added before other middleware to capture context)
app.add_middleware(LoggingContextMiddleware)

# Add custom metrics middleware
app.add_middleware(MetricsMiddleware)

# Add profiling middleware (DEBUG only — saves HTML to /tmp on slow requests)
if settings.DEBUG:
    app.add_middleware(ProfilingMiddleware)

# Add correlation ID middleware — must be outermost so request_id is set before all others
app.add_middleware(CorrelationIdMiddleware)

# Set up rate limiter exception handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # pyright: ignore[reportArgumentType]


# Add validation exception handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors from request data.

    Args:
        request: The request that caused the validation error
        exc: The validation error

    Returns:
        JSONResponse: A formatted error response
    """
    # Log the validation error
    logger.error(
        "validation_error",
        client_host=request.client.host if request.client else "unknown",
        path=request.url.path,
        errors=str(exc.errors()),
    )

    # Format the errors to be more user-friendly
    formatted_errors = []
    for error in exc.errors():
        loc = " -> ".join([str(loc_part) for loc_part in error["loc"] if loc_part != "body"])
        formatted_errors.append({"field": loc, "message": error["msg"]})

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Validation error", "errors": formatted_errors},
    )


# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)

# Serve the agent to other agents over A2A at /a2a (card at /a2a/.well-known/agent-card.json).
if settings.A2A_ENABLED:
    from app.services.a2a.server import (  # optional "a2a" extra
        agent_ask,
        build_a2a_app,
        build_agent_card,
        parse_caller_tokens,
    )

    app.mount(
        "/a2a",
        build_a2a_app(
            build_agent_card(
                name=settings.PROJECT_NAME,
                description=settings.A2A_AGENT_DESCRIPTION,
                url=settings.A2A_PUBLIC_URL,
                skills=[
                    {
                        "id": "chat",
                        "name": "Answer questions",
                        "description": settings.A2A_AGENT_DESCRIPTION,
                        "tags": ["qa"],
                    }
                ],
            ),
            agent_ask(agent),
            token=settings.A2A_SERVER_TOKEN,
            callers=parse_caller_tokens(settings.A2A_SERVER_TOKENS),
        ),
    )


@app.get("/")
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["root"][0])
async def root(request: Request):
    """Root endpoint returning basic API information."""
    logger.info("root_endpoint_called")
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "healthy",
        "environment": settings.ENVIRONMENT.value,
        "swagger_url": "/docs",
        "redoc_url": "/redoc",
    }


@app.get("/health")
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["health"][0])
async def health_check(request: Request) -> JSONResponse:
    """Health check endpoint with environment-specific information.

    Returns:
        JSONResponse: Health status payload, with HTTP 503 when the
        database is unreachable so load balancers can drop the instance.
    """
    logger.info("health_check_called")

    # Check database connectivity
    db_healthy = await database_service.health_check()

    response = {
        "status": "healthy" if db_healthy else "degraded",
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT.value,
        "components": {"api": "healthy", "database": "healthy" if db_healthy else "unhealthy"},
        "timestamp": datetime.now().isoformat(),
    }

    # If DB is unhealthy, set the appropriate status code
    status_code = status.HTTP_200_OK if db_healthy else status.HTTP_503_SERVICE_UNAVAILABLE

    return JSONResponse(content=response, status_code=status_code)
