class MockOpenAI:
    def chat_completion_create(self, model, messages, temperature=0.7):
        # Mock response
        system_message = next(
            (m for m in messages if m["role"] == "system"), None)
        user_message = next((m for m in messages if m["role"] == "user"), None)

        response_content = f"This is a mock response. I would analyze the text based on: {system_message['content'] if system_message else 'No system prompt'}"

        if "anomaly" in user_message["content"].lower() or "unusual" in user_message["content"].lower():
            response_content += "\n\nClassification: ANOMALOUS\nExplanation: This clause contains unusual terms that may be unfair to users."
        else:
            response_content += "\n\nClassification: NORMAL\nExplanation: This clause appears to be standard legal language."

        return MockResponse(response_content)


class MockResponse:
    def __init__(self, content):
        self.choices = [MockChoice(content)]


class MockChoice:
    def __init__(self, content):
        self.message = MockMessage(content)


class MockMessage:
    def __init__(self, content):
        self.content = content


# Use this instead of OpenAI in your code for testing
mock_openai = MockOpenAI()
