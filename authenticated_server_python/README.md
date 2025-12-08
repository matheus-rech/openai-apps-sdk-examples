# Ecommerce MCP server (Python)

This server exposes a single tool that renders the ecommerce widget hydrated by
sample catalog data. The tool performs a simple text search over
`sample_data.json` and returns matching products as `cartItems`, allowing the
widget to display the results without any hard-coded inventory.

## Prerequisites

- Python 3.10+
- A virtual environment (recommended)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the server

```bash
python main.py
```

The server listens on `http://127.0.0.1:8000` and exposes the standard MCP
endpoints:

- `GET /mcp` for the SSE stream
- `POST /mcp/messages?sessionId=...` for follow-ups

The `ecommerce-search` tool uses the `searchTerm` argument to filter products
by name, description, tags, or highlights and returns structured content with
`cartItems` so the ecommerce widget can hydrate itself.

## Customization

- Update `sample_data.json` with your own products.
- Adjust the search logic in `main.py` to match your catalog rules.
- Rebuild the widget assets (`pnpm run build`) if you change the UI.
