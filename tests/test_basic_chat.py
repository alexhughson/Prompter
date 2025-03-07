from prompter.schemas import Prompt, TextMessage


def test_basic_chat(llm_executor):
    """Test basic chat functionality without any special features"""
    prompt = Prompt(
        system_message="You are a helpful assistant that always responds with 'YES' or 'NO'.  Respond with no other text than yes or no",
        messages=[
            TextMessage.user("Is the sky blue?"),
        ],
    )

    response = llm_executor.execute(prompt)
    response.raise_for_status()

    # The response should be either YES or NO due to system message
    assert response.text().strip().upper() in ["YES", "NO"]

    for message in response.messages():
        assert message.content in ["YES", "NO"]
