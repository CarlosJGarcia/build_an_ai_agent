# Execute MCP Inspector
npx @modelcontextprotocol/inspector env NODE_TLS_REJECT_UNAUTHORIZED=0 TAVILY_API_KEY=$TAVILY_API_KEY npx -y tavily-mcp@latest

# Run this in the client, then open Safari and open http://127.0.0.1:6274
# ssh -L 6274:localhost:6274 user@workstation.url
