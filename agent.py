import asyncio
from dotenv import load_dotenv
load_dotenv()
from google.antigravity import Agent, LocalAgentConfig

async def main():
    # Configure the agent with a persona
    config = LocalAgentConfig(
        system_instructions="You are an expert assistant for codebase navigation."
    )
    
    # The async context manager handles the lifecycle automatically
    async with Agent(config) as agent:
        response = await agent.chat("What files are in the current directory?")
        print(await response.text())

if __name__ == "__main__":
    asyncio.run(main())