import os
import chainlit as cl
from agents import Agent, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, Runner
from dotenv import load_dotenv, find_dotenv
from openai.types.responses import ResponseTextDeltaEvent

# Load environment variables
load_dotenv(find_dotenv())

# Get Gemini API key (replace with your actual key or keep it in .env)
gemini_api_key = os.getenv("GEMINI_API_KEY") 

# Set up the provider
provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Define the model
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=provider,
)

# Run configuration
run_config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True,
)

# Define the Areeba Irfan Agent with specific instructions
agent1 = Agent(
    instructions=(
    "You are the Areeba Irfan Agent, a helpful assistant restricted to answering questions about: "
    "who Areeba Irfan can contact for professional networking or collaborations, how to arrange a meeting with Areeba Irfan, "
    "what Areeba Irfan does in her free time (e.g., professional development or personal hobbies), "
    "Areeba Irfan’s career history, technical or professional skills, and past or current projects. "
    "Politely decline to answer any questions outside these topics, stating that you are only authorized to provide information "
    "related to Areeba Irfan’s professional connections, skills, projects, and personal interests."
    ),
    name="Areeba Irfan Agent",
)

# Function to check if the question is relevant
def is_relevant_question(question: str) -> bool:
    """
    Check if the user's question is related to the allowed topics.
    This is a simple keyword-based approach; you can enhance it with NLP if needed.
    """
    question = question.lower()
    relevant_keywords = [
        "contact", "meet", "free time", "hobby", "hobbies", "past experience",
        "experience", "skill", "skills", "project", "projects", "who", "how", "what"
    ]
    return any(keyword in question for keyword in relevant_keywords)

@cl.on_chat_start
async def handle_chat_start():
    # Initialize chat history
    cl.user_session.set("history", [])
    await cl.Message(content="Hello! I'm the Areeba Irfan Agent.How can I assist you?").send()

@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get("history")
    
    # Create a message object for streaming
    msg = cl.Message(content="")
    await msg.send()

    # Check if the question is relevant
    if not is_relevant_question(message.content):
        response = "I'm sorry, I can only answer questions about who Areeba Irfan can contact, how to meet them, what they do in their free time, their past experience, skills, and projects. Please ask a relevant question."
        await msg.stream_token(response)
        history.append({"role": "user", "content": message.content})
        history.append({"role": "assistant", "content": response})
        cl.user_session.set("history", history)
        return

    # Append user message to history
    history.append({"role": "user", "content": message.content})

    # Run the agent with the input history
    result = Runner.run_streamed(
        agent1,
        input=history,
        run_config=run_config,
    )

    # Stream the response
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            await msg.stream_token(event.data.delta)

    # Append the final assistant response to history
    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("history", history)