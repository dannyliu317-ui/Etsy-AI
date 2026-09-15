"""
Etsy AI V3 - Image Generator

Image generation engine for Etsy jewelry photography.

Supported modes:
- Product Only Hero
- Background Replacement
- Lifestyle Scene
- Model Wearing

The original product image is always treated as the source of truth.
"""

import base64
import os
from typing import List, Optional

from openai import OpenAI


SUPPORTED_MODES = [
    "Product Only Hero",
    "Background Replacement",
    "Lifestyle Scene",
    "Model Wearing",
]


def get_client() -> OpenAI:
    """
    Create the OpenAI client from the environment variable.

    Never hard-code the API key in source code.
    """

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured."
        )

    return OpenAI(api_key=api_key)


def build_generation_prompt(
    image_prompt: str,
    mode: str,
) -> str:
    """
    Add final image-generation constraints to the
    prompt produced by image_studio.py.
    """

    if mode not in SUPPORTED_MODES:
        raise ValueError(
            f"Unsupported image mode: {mode}"
        )

    return f"""
You are creating a professional commercial jewelry photograph
for an Etsy US jewelry shop.

SOURCE PRODUCT PROTECTION IS THE HIGHEST PRIORITY.

The uploaded image contains the actual jewelry product.

Preserve the exact physical product shown in the source image.

Do NOT:
- redesign the jewelry
- change its shape
- change proportions
- change gemstone color
- change gemstone pattern
- add gemstones
- remove gemstones
- change metal color
- change metal finish
- change engraving
- change chain structure
- change clasp
- change thickness
- change stone placement
- invent details

The result must look like the SAME physical jewelry photographed
in a professionally controlled environment.

IMAGE MODE:
{mode}

SCENE INSTRUCTIONS:
{image_prompt}

PHOTOGRAPHY QUALITY:

Create photorealistic commercial jewelry photography.

Use:
- physically believable global illumination
- realistic environmental reflections
- natural light bounce
- realistic contact shadows
- accurate metal reflections
- realistic gemstone light response
- subtle micro-surface texture
- natural depth of field
- premium editorial photography

Use a warm primary light and a subtle cool secondary/rim light
when appropriate to the scene.

Avoid:
- fake HDR
- excessive glow
- plastic-looking metal
- unrealistic gemstone shine
- artificial halos
- harsh cutout edges
- floating products
- distorted jewelry
- excessive props

COMPOSITION:

1:1 square composition.

The jewelry should be the primary visual subject.

Optimize the composition for Etsy thumbnail viewing.

Keep approximately 60–80% of the meaningful visual attention
on the jewelry while maintaining tasteful negative space.

Do not add text, logos, watermarks, labels, or typography.

Return a polished commercial photograph.
""".strip()


def image_to_data_url(
    image_bytes: bytes,
    mime_type: str = "image/png",
) -> str:
    """
    Convert image bytes to a data URL.
    """

    encoded = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


def generate_images(
    image_bytes: bytes,
    image_prompt: str,
    mode: str,
    mime_type: str = "image/png",
    number_of_images: int = 1,
) -> List[bytes]:
    """
    Generate one or more candidate images.

    The exact image API parameters may vary by the installed
    OpenAI SDK/model version, so this function keeps the API
    boundary isolated from the Streamlit application.
    """

    if not image_bytes:
        raise ValueError(
            "Source image is empty."
        )

    if mode not in SUPPORTED_MODES:
        raise ValueError(
            f"Unsupported image mode: {mode}"
        )

    if number_of_images < 1:
        raise ValueError(
            "number_of_images must be at least 1."
        )

    client = get_client()

    final_prompt = build_generation_prompt(
        image_prompt=image_prompt,
        mode=mode,
    )

    source_data_url = image_to_data_url(
        image_bytes=image_bytes,
        mime_type=mime_type,
    )

    generated_images: List[bytes] = []

    for _ in range(number_of_images):

        try:

            result = client.images.edit(
                model="gpt-image-1",
                image=[
                    {
                        "image": source_data_url,
                    }
                ],
                prompt=final_prompt,
                size="1024x1024",
            )

            if not result.data:
                raise RuntimeError(
                    "The image generation API returned no image."
                )

            image_data = result.data[0]

            if getattr(
                image_data,
                "b64_json",
                None,
            ):

                generated_images.append(
                    base64.b64decode(
                        image_data.b64_json
                    )
                )

            elif getattr(
                image_data,
                "url",
                None,
            ):

                raise RuntimeError(
                    "The API returned a remote image URL. "
                    "URL downloading is intentionally handled "
                    "outside this module."
                )

            else:

                raise RuntimeError(
                    "The API returned an unsupported image format."
                )

        except Exception as error:

            raise RuntimeError(
                f"Image generation failed: {error}"
            ) from error

    return generated_images


def save_image(
    image_bytes: bytes,
    file_path: str,
) -> str:
    """
    Save generated image bytes to disk.
    """

    if not image_bytes:
        raise ValueError(
            "Cannot save an empty image."
        )

    with open(
        file_path,
        "wb",
    ) as image_file:

        image_file.write(
            image_bytes
        )

    return file_path
