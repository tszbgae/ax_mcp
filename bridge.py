import asyncio
import os
import sys
from typing import Optional

# Third-party imports
import ollama
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# CONFIGURATION
OLLAMA_MODEL = "qwen38B_analyst:latest" # The model we created in step 2
MCP_SERVER_SCRIPT = "server.py"  # Your Ax MCP server file

class AxOllamaBridge:
    def __init__(self):
        self.history = []

    async def run(self):
        # 1. DEFINE SERVER CONNECTION
        server_params = StdioServerParameters(
            command="uv", # Or "python" if not using uv
            args=["run", MCP_SERVER_SCRIPT], 
            env=os.environ.copy(),
        )

        # 2. CONNECT TO MCP SERVER
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # 3. DISCOVER TOOLS
                tools_list = await session.list_tools()
                print(f"🔗 Connected to Ax Server. Loaded {len(tools_list.tools)} tools.")
                
                # 4. CONVERT MCP TOOLS TO OLLAMA FORMAT
                ollama_tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema
                        }
                    }
                    for tool in tools_list.tools
                ]

                # 5. START CHAT LOOP
                print(f"🤖 Agent '{OLLAMA_MODEL}' ready. Type 'quit' to exit.")
                while True:
                    user_input = input("\n👤 You: ")
                    if user_input.lower() in ["quit", "exit"]:
                        break

                    self.history.append({"role": "user", "content": user_input})
                    await self.process_turn(session, ollama_tools)

    async def process_turn(self, session, tools):
        # Send history + tools to Ollama
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=self.history,
            tools=tools,
        )

        message = response['message']
        self.history.append(message)

        # CHECK FOR TOOLS
        if not message.get('tool_calls'):
            # No tools called, just print the text response
            print(f"\n🤖 Scientist: {message['content']}")
            return

        # EXECUTE TOOLS
        print(f"\n⚙️  Model requested {len(message['tool_calls'])} tool(s)...")
        
        for tool_call in message['tool_calls']:
            fn_name = tool_call['function']['name']
            fn_args = tool_call['function']['arguments']
            
            print(f"   > Executing: {fn_name}({fn_args})")
            
            try:
                # Call the MCP Server
                result = await session.call_tool(fn_name, arguments=fn_args)
                
                # Create the tool result message for Ollama
                # Note: Ollama expects the role "tool" 
                tool_msg = {
                    "role": "tool",
                    "content": str(result.content),
                    "name": fn_name,
                }
                self.history.append(tool_msg)
                print(f"   < Result: {str(result.content)[:100]}...") # Truncate log
                
            except Exception as e:
                error_msg = f"Error executing {fn_name}: {str(e)}"
                print(f"   ! {error_msg}")
                self.history.append({
                    "role": "tool", 
                    "content": error_msg, 
                    "name": fn_name
                })

        # RECURSIVE CALL: Give results back to Ollama to interpret
        await self.process_turn(session, tools)

if __name__ == "__main__":
    try:
        asyncio.run(AxOllamaBridge().run())
    except KeyboardInterrupt:
        print("\nBridge closed.")