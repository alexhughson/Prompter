import pytest
from pydantic import BaseModel

from prompter.schemas import Prompt, TextMessage


# class Character(BaseModel):
#     name: str
#     species: str
#     age: int


# def test_json_response_format(llm_executor):
#     """Test JSON response format with schema"""
#     prompt = Prompt(
#         system_message="You are a character creator",
#         messages=[TextMessage.user("Create a sci-fi character")],
#     )

#     response = llm_executor.execute_formatted(prompt, Character)
#     response.raise_for_status()

#     # Validate schema result
#     result = response.result()
#     assert result.valid()

#     # Parse into Character instance
#     character = result.parse()
#     assert isinstance(character, Character)
#     assert character.name
#     assert character.species
#     assert isinstance(character.age, int)


# def test_invalid_json_response():
#     """Test handling of invalid JSON response"""
#     executor = StubExecutor(invalid_json=True)

#     prompt = Prompt(
#         system_message="You are a character creator",
#         messages=[UserMessage("Create a sci-fi character")],
#         response_schema=Character,
#     )

#     response = executor.execute(prompt)
#     result = response.result()

#     assert not result.valid()
#     assert not result.valid_json()

#     with pytest.raises(Exception):
#         result.parse()


# def test_schema_mismatch_response():
#     """Test handling of JSON that doesn't match schema"""
#     executor = StubExecutor(invalid_schema=True)

#     prompt = Prompt(
#         system_message="You are a character creator",
#         messages=[UserMessage("Create a sci-fi character")],
#         response_schema=Character,
#     )

#     response = executor.execute(prompt)
#     result = response.result()

#     assert result.valid_json()  # Valid JSON
#     assert not result.valid()  # But doesn't match schema

#     # Can get raw dict
#     data = result.parse_obj()
#     assert isinstance(data, dict)

#     # But schema parsing fails
#     with pytest.raises(Exception):
#         result.parse()
