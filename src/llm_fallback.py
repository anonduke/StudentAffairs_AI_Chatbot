import os
import openai   
def ask_llm(question, context_buffer, system_prompt=None):
   
    if api_key is None:
        return "LLM not configured. Please set the OpenAI API key."

    client = openai.OpenAI(api_key=api_key)

    if not system_prompt:
        system_prompt = (
            "You are a student affairs chatbot. Answer questions about college life, fees, policies, academics, "
            "and student support. Be concise and helpful. If the question is not about student affairs, politely say so."
            "here the context is: " + str(context_buffer)
        )
        print(f"Using system prompt: {system_prompt}")
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            temperature=0.4,
            max_tokens=400,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"(LLM error: {e})"