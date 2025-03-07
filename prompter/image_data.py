import base64
from dataclasses import dataclass

import httpx


@dataclass
class ImageData:
    base64_data: str
    content_type: str


def url_to_b64(url: str) -> ImageData:
    response = httpx.get(url)
    content_type = response.headers.get("content-type")
    base64_data = base64.standard_b64encode(response.content).decode("utf-8")
    return ImageData(base64_data=base64_data, content_type=content_type)
