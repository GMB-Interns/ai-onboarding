import chainlit as cl
import openai
import os
from langchain import PromptTemplate, OpenAI, LLMChain
import getpass

# Get the API key securely from the user
api_key = getpass.getpass("Enter your OpenAI API Key: ")

# Set the environment variable and openai library attribute
os.environ['OPENAI_API_KEY'] = api_key
openai.api_key = api_key



BEHAVIORAL_PROMPT = """
I'm a behavioral plan designer, focusing on Tiny Habits and self-determination theory. Help me draft a step-by-step plan based on the following details:

- Plan Timeframe: Specific period, week number, start and end dates.
- Audience: Should be tailored for a specific audience.
- Weekly Goal: Articulated goals that cumulate to the overall goal.
- Task Breakdown: Large tasks should be split into smaller, achievable tasks.
- Daily Plan Format: 
  | Week | Day | Day Number | Day of the Week | Behavior | Category | Specifics | Time | Duration | Location | Prompts | Milestones |
- Behavior: Commands to give clear imagery of activities.
- Category: Health & Fitness, Work, Learn, or Life.
- Specifics: Include all necessary details.
- Prompts: 1-3 questions that help evaluate the plan's progress. Avoid Yes/No questions.

To customize the plan, I need answers to:
1. Topic: What's the plan's focus?
2. Goal: The desired outcome?
3. Type: Individual or group plan?
4. Target Audience: Who will execute the plan? Any specific requirements for them?
5. Plan Duration: Desired length, up to 12 weeks.
6. Familiarity: Current competency level with the topic.
7. Daily Time: Amount of time available each day.
8. Preferred Time: Any specific time of the day?
9. Challenge Level: Scale of 1-10.
10. Resources: Any pre-existing sources?

Kindly provide one answer at a time.

Question: {question}
"""

@cl.on_chat_start
def chat_start():
    prompt = PromptTemplate(template=BEHAVIORAL_PROMPT, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=OpenAI(temperature=0, streaming=True), verbose=True)

    cl.user_session.set("llm_chain", llm_chain)

@cl.on_message
async def handle_message(message: str):
    llm_chain = cl.user_session.get("llm_chain")

    res = await llm_chain.acall({"question": message}, callbacks=[cl.AsyncLangchainCallbackHandler()])

    await cl.Message(content=res["text"]).send()










