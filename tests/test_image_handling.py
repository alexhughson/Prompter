from prompter.schemas import ImageMessage, Prompt, TextMessage

# Test images URLs (we should probably host these somewhere reliable)
CLOUD_IMAGE = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Cumulus_humilis_clouds_in_Ukraine.jpg/1920px-Cumulus_humilis_clouds_in_Ukraine.jpg"
CAT_IMAGE = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"


def test_image_description(llm_executor):
    """Test that the LLM can correctly describe images"""
    prompt = Prompt(
        system_message="You are a helpful assistant. Describe what you see in the image in one sentence.",
        messages=[
            TextMessage.user("What is in this image?"),
            ImageMessage.user(url=CLOUD_IMAGE),
        ],
    )

    response = llm_executor.execute(prompt)
    response.raise_for_status()

    text = response.text().lower()
    assert any(word in text for word in ["cloud", "sky", "cumulus"])


def test_image_with_question(llm_executor):
    """Test that the LLM can answer specific questions about images"""
    prompt = Prompt(
        system_message="You are a helpful assistant.",
        messages=[
            ImageMessage.user(url=CAT_IMAGE),
            TextMessage.user(content="What color is this animal?"),
        ],
    )

    response = llm_executor.execute(prompt)
    response.raise_for_status()

    # The cat in the image should be described as some variation of orange/ginger
    text = response.text().lower()
    assert any(word in text for word in ["orange", "ginger", "red"])
