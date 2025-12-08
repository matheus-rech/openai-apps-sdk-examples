"""MCP server for an authenticated app implemented with the Python FastMCP helper."""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

import mcp.types as types
from mcp.server.fastmcp import FastMCP
from mcp.shared.auth import ProtectedResourceMetadata
from pydantic import AnyHttpUrl
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


@dataclass(frozen=True)
class PizzazWidget:
    identifier: str
    title: str
    template_uri: str
    invoking: str
    invoked: str
    html: str
    response_text: str


ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


@lru_cache(maxsize=None)
def _load_widget_html(component_name: str) -> str:
    html_path = ASSETS_DIR / f"{component_name}.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf8")

    fallback_candidates = sorted(ASSETS_DIR.glob(f"{component_name}-*.html"))
    if fallback_candidates:
        return fallback_candidates[-1].read_text(encoding="utf8")

    raise FileNotFoundError(
        f'Widget HTML for "{component_name}" not found in {ASSETS_DIR}. '
        "Run `pnpm run build` to generate the assets before starting the server."
    )


CAROUSEL_WIDGET = PizzazWidget(
    identifier="pizza-carousel",
    title="Show Pizza Carousel",
    template_uri="ui://widget/pizza-carousel.html",
    invoking="Carousel some spots",
    invoked="Served a fresh carousel",
    html=_load_widget_html("pizzaz-carousel"),
    response_text="Rendered a pizza carousel!",
)


MIME_TYPE = "text/html+skybridge"

SEARCH_TOOL_NAME = CAROUSEL_WIDGET.identifier

SEARCH_TOOL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "title": "Search terms",
    "properties": {
        "searchTerm": {
            "type": "string",
            "description": "Optional text to echo back in the response.",
        },
    },
    "required": [],
    "additionalProperties": False,
}

DEFAULT_AUTH_SERVER_URL = "https://dev-65wmmp5d56ev40iy.us.auth0.com/"
DEFAULT_RESOURCE_SERVER_URL = "http://localhost:8000/mcp"

# Public URLs that describe this resource server plus the authorization server.
AUTHORIZATION_SERVER_URL = AnyHttpUrl(
    os.environ.get("AUTHORIZATION_SERVER_URL", DEFAULT_AUTH_SERVER_URL)
)
RESOURCE_SERVER_URL = "https://945c890ee720.ngrok-free.app"
RESOURCE_SCOPES = []

_parsed_resource_url = urlparse(str(RESOURCE_SERVER_URL))
_resource_path = (
    _parsed_resource_url.path if _parsed_resource_url.path not in ("", "/") else ""
)
PROTECTED_RESOURCE_METADATA_PATH = (
    f"/.well-known/oauth-protected-resource{_resource_path}"
)
PROTECTED_RESOURCE_METADATA_URL = f"{_parsed_resource_url.scheme}://{_parsed_resource_url.netloc}{PROTECTED_RESOURCE_METADATA_PATH}"

print("PROTECTED_RESOURCE_METADATA_URL", PROTECTED_RESOURCE_METADATA_URL)
PROTECTED_RESOURCE_METADATA = ProtectedResourceMetadata(
    resource=RESOURCE_SERVER_URL,
    authorization_servers=[AUTHORIZATION_SERVER_URL],
    scopes_supported=RESOURCE_SCOPES,
)

# Tool-level securitySchemes inform ChatGPT when OAuth is required for a call.
MIXED_TOOL_SECURITY_SCHEMES = [
    {"type": "noauth"},
    {
        "type": "oauth2",
        "scopes": RESOURCE_SCOPES,
    },
]


mcp = FastMCP(
    name="pizzaz-python",
    stateless_http=True,
)


def _resource_metadata_url() -> str | None:
    auth_config = getattr(mcp.settings, "auth", None)
    if auth_config and auth_config.resource_server_url:
        parsed = urlparse(str(auth_config.resource_server_url))
        resource_path = parsed.path if parsed.path not in ("", "/") else ""
        return f"{parsed.scheme}://{parsed.netloc}/.well-known/oauth-protected-resource{resource_path}"
    print("PROTECTED_RESOURCE_METADATA_URL", PROTECTED_RESOURCE_METADATA_URL)
    return PROTECTED_RESOURCE_METADATA_URL


def _build_www_authenticate_value(error: str, description: str) -> str:
    safe_error = error.replace('"', r"\"")
    safe_description = description.replace('"', r"\"")
    parts = [
        f'error="{safe_error}"',
        f'error_description="{safe_description}"',
    ]
    resource_metadata = _resource_metadata_url()
    if resource_metadata:
        parts.append(f'resource_metadata="{resource_metadata}"')
    return f"Bearer {', '.join(parts)}"


def _oauth_error_result(
    user_message: str,
    *,
    error: str = "invalid_request",
    description: str | None = None,
) -> types.ServerResult:
    description_text = description or user_message
    return types.ServerResult(
        types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=user_message,
                )
            ],
            _meta={
                "mcp/www_authenticate": [
                    _build_www_authenticate_value(error, description_text)
                ]
            },
            isError=True,
        )
    )


def _get_bearer_token_from_request() -> str | None:
    try:
        request_context = mcp._mcp_server.request_context
    except LookupError:
        return None

    request = getattr(request_context, "request", None)
    if request is None:
        return None

    header_value: Any = None
    headers = getattr(request, "headers", None)
    if headers is not None:
        try:
            header_value = headers.get("authorization")
            if header_value is None:
                header_value = headers.get("Authorization")
        except Exception:
            header_value = None

    if header_value is None:
        # Attempt to read from ASGI scope headers if available
        scope = getattr(request, "scope", None)
        scope_headers = scope.get("headers") if isinstance(scope, dict) else None
        if scope_headers:
            for key, value in scope_headers:
                decoded_key = (
                    key.decode("latin-1")
                    if isinstance(key, (bytes, bytearray))
                    else str(key)
                ).lower()
                if decoded_key == "authorization":
                    header_value = (
                        value.decode("latin-1")
                        if isinstance(value, (bytes, bytearray))
                        else str(value)
                    )
                    break

    if header_value is None and isinstance(request, dict):
        # Fall back to dictionary-like request contexts
        raw_value = request.get("authorization") or request.get("Authorization")
        header_value = raw_value

    if header_value is None:
        return None

    if isinstance(header_value, (bytes, bytearray)):
        header_value = header_value.decode("latin-1")

    header_value = header_value.strip()
    if not header_value.lower().startswith("bearer "):
        return None

    token = header_value[7:].strip()
    return token or None


@mcp.custom_route(PROTECTED_RESOURCE_METADATA_PATH, methods=["GET", "OPTIONS"])
async def protected_resource_metadata(request: Request) -> Response:
    """Expose RFC 9728 metadata so clients can find the Auth0 authorization server."""
    if request.method == "OPTIONS":
        return Response(status_code=204)
    return JSONResponse(PROTECTED_RESOURCE_METADATA.model_dump(mode="json"))


def _resource_description(widget: PizzazWidget) -> str:
    return f"{widget.title} widget markup"


def _tool_meta(
    widget: PizzazWidget,
    security_schemes: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    meta = {
        "openai/outputTemplate": widget.template_uri,
        "openai/toolInvocation/invoking": widget.invoking,
        "openai/toolInvocation/invoked": widget.invoked,
        "openai/widgetAccessible": True,
        "openai/resultCanProduceWidget": True,
    }
    if security_schemes is not None:
        meta["securitySchemes"] = deepcopy(security_schemes)
    return meta


def _tool_invocation_meta(widget: PizzazWidget) -> Dict[str, Any]:
    return {
        "openai/toolInvocation/invoking": widget.invoking,
        "openai/toolInvocation/invoked": widget.invoked,
        "openai/widgetSessionId": "ren-test-session-id",
    }


def _tool_error(message: str) -> types.ServerResult:
    return types.ServerResult(
        types.CallToolResult(
            content=[types.TextContent(type="text", text=message)],
            isError=True,
        )
    )


@mcp._mcp_server.list_tools()
async def _list_tools() -> List[types.Tool]:
    tool_meta = _tool_meta(CAROUSEL_WIDGET, MIXED_TOOL_SECURITY_SCHEMES)
    return [
        types.Tool(
            name=SEARCH_TOOL_NAME,
            title=CAROUSEL_WIDGET.title,
            description="Echo the provided search terms.",
            inputSchema=SEARCH_TOOL_SCHEMA,
            _meta=tool_meta,
            securitySchemes=list(MIXED_TOOL_SECURITY_SCHEMES),
            # To disable the approval prompt for the tools
            annotations={
                "destructiveHint": False,
                "openWorldHint": False,
                "readOnlyHint": True,
            },
        ),
    ]


@mcp._mcp_server.list_resources()
async def _list_resources() -> List[types.Resource]:
    return [
        types.Resource(
            name=CAROUSEL_WIDGET.title,
            title=CAROUSEL_WIDGET.title,
            uri=CAROUSEL_WIDGET.template_uri,
            description=_resource_description(CAROUSEL_WIDGET),
            mimeType=MIME_TYPE,
            _meta=_tool_meta(CAROUSEL_WIDGET),
        )
    ]


@mcp._mcp_server.list_resource_templates()
async def _list_resource_templates() -> List[types.ResourceTemplate]:
    return [
        types.ResourceTemplate(
            name=CAROUSEL_WIDGET.title,
            title=CAROUSEL_WIDGET.title,
            uriTemplate=CAROUSEL_WIDGET.template_uri,
            description=_resource_description(CAROUSEL_WIDGET),
            mimeType=MIME_TYPE,
            _meta=_tool_meta(CAROUSEL_WIDGET),
        )
    ]


async def _handle_read_resource(req: types.ReadResourceRequest) -> types.ServerResult:
    requested_uri = str(req.params.uri)
    if requested_uri != CAROUSEL_WIDGET.template_uri:
        return types.ServerResult(
            types.ReadResourceResult(
                contents=[],
                _meta={"error": f"Unknown resource: {req.params.uri}"},
            )
        )

    contents = [
        types.TextResourceContents(
            uri=CAROUSEL_WIDGET.template_uri,
            mimeType=MIME_TYPE,
            text=CAROUSEL_WIDGET.html,
            _meta=_tool_meta(CAROUSEL_WIDGET),
        )
    ]

    return types.ServerResult(types.ReadResourceResult(contents=contents))


async def _call_tool_request(req: types.CallToolRequest) -> types.ServerResult:
    tool_name = req.params.name
    if tool_name != SEARCH_TOOL_NAME:
        return _tool_error(f"Unknown tool: {req.params.name}")

    arguments = req.params.arguments or {}
    meta = _tool_invocation_meta(CAROUSEL_WIDGET)
    topping = str(arguments.get("searchTerm", "")).strip()

    if not _get_bearer_token_from_request():
        return _oauth_error_result(
            "Authentication required: no access token provided.",
            description="No access token was provided",
        )

    return types.ServerResult(
        types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text="Rendered a pizza carousel!",
                )
            ],
            structuredContent={"pizzaTopping": topping},
            _meta=meta,
        )
    )


mcp._mcp_server.request_handlers[types.CallToolRequest] = _call_tool_request
mcp._mcp_server.request_handlers[types.ReadResourceRequest] = _handle_read_resource


app = mcp.streamable_http_app()

try:
    from starlette.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
    )
except Exception:
    pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000)
