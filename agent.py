"""
Vertex - Intelligent Multi-Channel Agent Orchestration
Built with LlamaIndex, OpenAI GPT-4o, and Python
"""

import os
from dotenv import load_dotenv

load_dotenv()

try:
    from llama_index import OpenAI
    from llama_index.tools import FunctionTool
    from llama_index.agent import OpenAIAgent

    print("All imports successful!")
except ImportError as e:
    print("Import error! Please run: pip install -r requirements.txt")
    print(f"Error details: {e}")
    exit(1)


def setup_agent():
    """Set up the AI agent with API keys and tools."""

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key or api_key == "your-openai-api-key-here":
        print("\nERROR: Please set your OpenAI API key in the .env file!")
        print("1. Open the .env file")
        print("2. Replace 'your-openai-api-key-here' with your actual API key")
        print("3. Get your API key from: https://platform.openai.com/api-keys")
        return None

    print("\nSetting up AI model...")
    llm = OpenAI(model="gpt-4o-mini", temperature=0.7, api_key=api_key)
    print("AI model configured!")

    return llm


def send_email(to: str, subject: str, body: str) -> str:
    """
    Send an email (currently in test mode).

    Args:
        to: Recipient email address
        subject: Email subject line
        body: Email body content

    Returns:
        Status message
    """
    print(f"\nEMAIL TOOL CALLED:")
    print(f"   To: {to}")
    print(f"   Subject: {subject}")
    print(f"   Body: {body[:100]}...")

    return f"[TEST MODE] Email would be sent to {to} with subject: '{subject}'"


email_tool = FunctionTool.from_defaults(
    fn=send_email,
    name="send_email",
    description="Send an email to someone. Needs: recipient email, subject, and message body."
)


def make_voice_call(phone: str, script: str) -> str:
    """
    Make a phone call and read a script (currently in test mode).

    Args:
        phone: Phone number (e.g., +14155550123)
        script: What to say during the call

    Returns:
        Call status
    """
    print(f"\nVOICE CALL TOOL CALLED:")
    print(f"   Phone: {phone}")
    print(f"   Script: {script[:100]}...")

    return f"[TEST MODE] Call would be placed to {phone}"


call_tool = FunctionTool.from_defaults(
    fn=make_voice_call,
    name="make_voice_call",
    description="Make a phone call and read a script. Needs: phone number and script text."
)


def calculate(expression: str) -> str:
    """
    Perform mathematical calculations.

    Args:
        expression: Math expression like "2 + 2" or "10 * 5"

    Returns:
        Result of calculation
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        print(f"\nCALCULATOR TOOL: {expression} = {result}")
        return f"The answer is: {result}"
    except Exception as e:
        return f"Cannot calculate that: {str(e)}"


calc_tool = FunctionTool.from_defaults(
    fn=calculate,
    name="calculate",
    description="Perform math calculations. Input a math expression like '2 + 2' or '100 * 5'."
)


def get_current_datetime() -> str:
    """Get the current date and time."""
    from datetime import datetime
    now = datetime.now()
    formatted = now.strftime("%A, %B %d, %Y at %I:%M %p")
    print(f"\nDATE/TIME TOOL: {formatted}")
    return f"Current date and time: {formatted}"


datetime_tool = FunctionTool.from_defaults(
    fn=get_current_datetime,
    name="get_datetime",
    description="Get the current date and time."
)


def create_agent(llm):
    """Create and return the AI agent with all tools."""

    tools = [email_tool, call_tool, calc_tool, datetime_tool]

    print("\nCreating AI Agent with tools:")
    for tool in tools:
        print(f"   * {tool.metadata.name}")

    agent = OpenAIAgent.from_tools(
        tools=tools,
        llm=llm,
        verbose=True
    )

    return agent


def main():
    """Run the AI agent."""

    print("=" * 70)
    print("Vertex - Intelligent Multi-Channel Orchestration")
    print("=" * 70)

    llm = setup_agent()
    if not llm:
        return

    agent = create_agent(llm)

    print("\n" + "=" * 70)
    print("Vertex is ready! You can ask it to:")
    print("   * Send emails (test mode)")
    print("   * Make phone calls (test mode)")
    print("   * Do math calculations")
    print("   * Tell you the current date/time")
    print("\nType 'quit' or 'exit' to stop")
    print("=" * 70 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ['exit', 'quit', 'q', 'bye']:
                print("\nGoodbye! Thanks for using Vertex!")
                break

            if not user_input:
                continue

            print("\nAgent is thinking...\n")

            response = agent.chat(user_input)

            print(f"\nAgent: {response}")
            print("\n" + "-" * 70 + "\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Try asking something else!\n")


if __name__ == "__main__":
    main()