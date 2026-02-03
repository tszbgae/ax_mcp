import asyncio
import os
import sys
import json
from llama_cpp import Llama

# MCP Imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# CONFIGURATION
MODEL_PATH = "models/qwen2.5-14b-instruct-q4_k_m.gguf"  # <--- UPDATE THIS PATH
MCP_SERVER_SCRIPT = "server.py"

class AxDirectRunner:
    def __init__(self):
        print(f"⏳ Loading model from {MODEL_PATH}...")
        # n_ctx=8192 is important for holding long experiment histories
        # n_gpu_layers=-1 attempts to offload ALL layers to GPU
        self.llm = Llama(
            model_path=MODEL_PATH,
            n_gpu_layers=-1, 
            n_ctx=8192,
            verbose=False
        )
        self.history = [
            {"role": "system", "content": "You are an autonomous scientist optimized for Ax experiments. Use the provided tools to run benchmarks."}
        ]
        print("✅ Model loaded.")

    async def run(self):
        # 1. Start the MCP Server
        server_params = StdioServerParameters(
            command="python", # or "uv" run
            args=[MCP_SERVER_SCRIPT],
            env=os.environ.copy(),
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # 2. Get Tools
                mcp_tools = await session.list_tools()
                
                # 3. Convert MCP Tools -> OpenAI Format (required by llama-cpp)
                openai_tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema
                        }
                    }
                    for tool in mcp_tools.tools
                ]
                
                print(f"🔗 Connected to MCP. Loaded {len(openai_tools)} tools.")
                print("🤖 Scientist ready. Type 'quit' to exit.")

                # 4. Main Chat Loop
                while True:
                    user_input = input("\n👤 You: ")
                    if user_input.lower() in ["quit", "exit"]:
                        break

                    self.history.append({"role": "user", "content": user_input})
                    
                    # Process the turn (with recursion for tool calls)
                    await self.process_turn(session, openai_tools)

    async def process_turn(self, session, tools):
        # Call the local model
        response = self.llm.create_chat_completion(
            messages=self.history,
            tools=tools,
            tool_choice="auto",
            temperature=0.1  # Low temp for rigorous science
        )

        message = response["choices"][0]["message"]
        self.history.append(message)

        # CHECK FOR TOOLS
        tool_calls = message.get("tool_calls")
        
        if not tool_calls:
            # No tools, just print text
            print(f"\n🤖 Scientist: {message['content']}")
            return

        # EXECUTE TOOLS
        print(f"\n⚙️  Model requested {len(tool_calls)} tool(s)...")
        
        for tool_call in tool_calls:
            fn_name = tool_call["function"]["name"]
            fn_args_str = tool_call["function"]["arguments"]
            
            # Parse arguments (Llama.cpp returns them as a string JSON)
            try:
                fn_args = json.loads(fn_args_str)
            except:
                fn_args = {}

            print(f"   > Executing: {fn_name}({fn_args_str})")

            try:
                # Call MCP Server
                result = await session.call_tool(fn_name, arguments=fn_args)
                result_content = str(result.content)

                # Append result to history (Role MUST be 'tool' for OpenAI format compatibility)
                self.history.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": fn_name,
                    "content": result_content
                })
                print(f"   < Result: {result_content[:100]}...")

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                print(f"   ! {error_msg}")
                self.history.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": fn_name,
                    "content": error_msg
                })

        # Recurse: Give the model a chance to read the tool results and respond
        await self.process_turn(session, tools)

if __name__ == "__main__":
    try:
        asyncio.run(AxDirectRunner().run())
    except KeyboardInterrupt:
        print("\nRunner closed.")