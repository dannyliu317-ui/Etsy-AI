import base64
import json
import os

import streamlit as st
from openai import OpenAI

from image_studio import (
    ProductProfile,
    build_image_prompt,
    get_scene_options,
)

from image_generator import generate_images


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Etsy AI",
    page_icon="💎",
    layout="wide",
)


# =========================================================
# HEADER
# =========================================================

st.title("💎 Etsy AI")

st.subheader(
    "AI Jewelry Listing & Image Studio"
)

st.write(
    "Upload jewelry images, analyze the product, "
    "generate Etsy SEO content, and create professional "
    "product photography prompts or images."
)


# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """
You are an expert Etsy SEO strategist, jewelry product analyst,
and American e-commerce copywriter for the US Etsy market.

Analyze only what is visible in the supplied product images
and what the seller explicitly provides.

Never invent:
- gemstone identity
- metal
- dimensions
- weight
- certification
- plating
- handmade status
- healing properties
- origin
- material composition
- unsupported product claims

If a fact is uncertain, describe only what is visibly supported
or leave it out.

Use natural American English.

Avoid keyword stuffing.

Create persuasive Etsy copy while remaining accurate.

Return valid JSON with exactly these keys:

product_analysis
seo_title
product_description
etsy_tags
search_keywords
buyer_benefits
gift_occasions

product_analysis must contain:

jewelry_type
materials
gemstone
colors
design_details
style
visual_positioning

etsy_tags must contain exactly 13 strings.

Every Etsy tag must be 20 characters or fewer.
"""


# =========================================================
# SIDEBAR - PRODUCT FACTS
# =========================================================

with st.sidebar:

    st.header("📌 Confirmed Product Facts")

    product_name = st.text_input(
        "Product Name",
        placeholder="Example: Jadeite Flower Stud Earrings",
    )

    material = st.text_input(
        "Material",
        placeholder="Example: 925 sterling silver",
    )

    gemstone = st.text_input(
        "Gemstone",
        placeholder="Example: Natural jadeite",
    )

    dimensions = st.text_input(
        "Size / Dimensions",
        placeholder="Example: 9.7 mm",
    )

    weight = st.text_input(
        "Weight",
        placeholder="Example: 1.22 g",
    )

    style = st.text_input(
        "Style",
        placeholder="Example: Elegant, vintage, dark nature",
    )

    extra_facts = st.text_area(
        "Other Confirmed Facts",
        placeholder="Add confirmed product information here.",
    )


# =========================================================
# IMAGE UPLOAD
# =========================================================

st.header("📷 1. Upload Product Images")

uploaded_files = st.file_uploader(
    "Upload one or more product images",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
    ],
    accept_multiple_files=True,
)


if uploaded_files:

    st.success(
        f"{len(uploaded_files)} product image(s) uploaded."
    )

    columns = st.columns(
        min(
            4,
            len(uploaded_files),
        )
    )

    for index, uploaded_file in enumerate(
        uploaded_files
    ):

        with columns[
            index % len(columns)
        ]:

            st.image(
                uploaded_file,
                caption=uploaded_file.name,
                use_container_width=True,
            )


# =========================================================
# ETSY LISTING GENERATOR
# =========================================================

if uploaded_files:

    st.header(
        "✨ 2. Generate Etsy Listing"
    )

    if st.button(
        "🚀 Analyze Product & Generate Etsy Listing",
        type="primary",
        use_container_width=True,
    ):

        if not os.getenv(
            "OPENAI_API_KEY"
        ):

            st.error(
                "OPENAI_API_KEY is not configured."
            )

            st.stop()


        client = OpenAI(
            api_key=os.environ[
                "OPENAI_API_KEY"
            ]
        )


        confirmed_information = f"""
Confirmed seller information:

Product name:
{product_name}

Material:
{material}

Gemstone:
{gemstone}

Dimensions:
{dimensions}

Weight:
{weight}

Style:
{style}

Other confirmed facts:
{extra_facts}
"""


        user_content = [

            {
                "type": "text",

                "text":
                    confirmed_information
                    +
                    """

Analyze all uploaded product images.

Generate:

1. Product Analysis
2. Etsy SEO Title
3. Product Description
4. Exactly 13 Etsy Tags
5. Search Keywords
6. Buyer Benefits
7. Gift Occasions
""",
            }
        ]


        # -------------------------------------------------
        # ADD PRODUCT IMAGES
        # -------------------------------------------------

        for uploaded_file in uploaded_files:

            encoded_image = base64.b64encode(
                uploaded_file.getvalue()
            ).decode("utf-8")


            mime_type = (
                uploaded_file.type
                or "image/jpeg"
            )


            user_content.append(

                {
                    "type": "image_url",

                    "image_url": {

                        "url":
                            f"data:{mime_type};base64,{encoded_image}"
                    },
                }
            )


        # -------------------------------------------------
        # AI ANALYSIS
        # -------------------------------------------------

        with st.spinner(
            "🔍 Analyzing jewelry and generating Etsy SEO..."
        ):

            try:

                response = (
                    client.chat.completions.create(

                        model="gpt-4.1-mini",

                        response_format={
                            "type": "json_object"
                        },

                        messages=[

                            {
                                "role": "system",
                                "content": SYSTEM_PROMPT,
                            },

                            {
                                "role": "user",
                                "content": user_content,
                            },
                        ],
                    )
                )


                result = json.loads(
                    response.choices[
                        0
                    ].message.content
                )


                st.session_state[
                    "etsy_listing"
                ] = result


                st.success(
                    "🎉 Etsy Listing generated successfully!"
                )


            except Exception as error:

                st.error(
                    f"Generation failed: {error}"
                )


# =========================================================
# ETSY LISTING RESULTS
# =========================================================

result = st.session_state.get(
    "etsy_listing"
)

analysis = {}


if result:

    st.divider()

    st.header(
        "🛍️ Etsy Listing Results"
    )


    # -----------------------------------------------------
    # PRODUCT ANALYSIS
    # -----------------------------------------------------

    st.subheader(
        "🔍 Product Analysis"
    )


    analysis = result.get(
        "product_analysis",
        {}
    )


    col1, col2 = st.columns(2)


    with col1:

        st.write(
            "**Jewelry Type**"
        )

        st.write(
            analysis.get(
                "jewelry_type",
                "",
            )
        )


        st.write(
            "**Material**"
        )

        st.write(
            analysis.get(
                "materials",
                [],
            )
        )


        st.write(
            "**Gemstone**"
        )

        st.write(
            analysis.get(
                "gemstone",
                "",
            )
        )


        st.write(
            "**Style**"
        )

        st.write(
            analysis.get(
                "style",
                "",
            )
        )


    with col2:

        st.write(
            "**Colors**"
        )

        st.write(
            analysis.get(
                "colors",
                [],
            )
        )


        st.write(
            "**Design Details**"
        )


        for detail in analysis.get(
            "design_details",
            [],
        ):

            st.write(
                f"• {detail}"
            )


        st.write(
            "**Visual Positioning**"
        )


        st.write(
            analysis.get(
                "visual_positioning",
                "",
            )
        )


    # -----------------------------------------------------
    # SEO TITLE
    # -----------------------------------------------------

    st.subheader(
        "🏷️ Etsy SEO Title"
    )


    st.text_area(

        "Title",

        result.get(
            "seo_title",
            "",
        ),

        height=100,

        key="seo_title_output",
    )


    # -----------------------------------------------------
    # DESCRIPTION
    # -----------------------------------------------------

    st.subheader(
        "📝 Product Description"
    )


    st.text_area(

        "Description",

        result.get(
            "product_description",
            "",
        ),

        height=350,

        key="description_output",
    )


    # -----------------------------------------------------
    # TAGS
    # -----------------------------------------------------

    st.subheader(
        "🔖 13 Etsy Tags"
    )


    tags = result.get(
        "etsy_tags",
        []
    )


    if len(tags) != 13:

        st.warning(
            f"AI returned {len(tags)} tags. "
            "Expected exactly 13."
        )


    tag_columns = st.columns(3)


    for index, tag in enumerate(tags):

        with tag_columns[
            index % 3
        ]:

            st.code(
                tag,
                language=None,
            )


    # -----------------------------------------------------
    # SEARCH KEYWORDS
    # -----------------------------------------------------

    st.subheader(
        "🔎 Search Keywords"
    )


    for keyword in result.get(
        "search_keywords",
        [],
    ):

        st.write(
            f"• {keyword}"
        )


    # -----------------------------------------------------
    # BUYER BENEFITS
    # -----------------------------------------------------

    st.subheader(
        "💎 Buyer Benefits"
    )


    for benefit in result.get(
        "buyer_benefits",
        [],
    ):

        st.write(
            f"• {benefit}"
        )


    # -----------------------------------------------------
    # GIFT OCCASIONS
    # -----------------------------------------------------

    st.subheader(
        "🎁 Gift Occasions"
    )


    for occasion in result.get(
        "gift_occasions",
        [],
    ):

        st.write(
            f"• {occasion}"
        )


    # -----------------------------------------------------
    # COMPLETE JSON
    # -----------------------------------------------------

    with st.expander(
        "📦 Complete Listing Data"
    ):

        st.json(
            result
        )


# =========================================================
# IMAGE STUDIO
# =========================================================

st.divider()

st.header(
    "🖼️ AI Image Studio"
)

st.write(
    "Create Etsy-ready jewelry photography "
    "while preserving the original product design."
)


# =========================================================
# IMAGE MODE
# =========================================================

image_mode = st.selectbox(

    "Image Mode",

    get_scene_options(),
)


# =========================================================
# NUMBER OF IMAGES
# =========================================================

number_of_images = st.selectbox(

    "Number of Images",

    [1, 2, 4],

    index=0,
)


scene_col1, scene_col2 = st.columns(2)


with scene_col1:

    st.info(
        f"Selected mode: {image_mode}"
    )


with scene_col2:

    st.write(
        "Scene style is automatically selected "
        "according to the product material, "
        "gemstone, style, and target customer."
    )


# =========================================================
# BUILD PRODUCT PROFILE
# =========================================================

profile = ProductProfile(

    jewelry_type=
        analysis.get(
            "jewelry_type",
            product_name,
        ),

    material=
        material
        or ", ".join(
            analysis.get(
                "materials",
                [],
            )
        ),

    gemstone=
        gemstone
        or analysis.get(
            "gemstone",
            "",
        ),

    dimensions=
        dimensions,

    style=
        style
        or analysis.get(
            "style",
            "",
        ),

    extra_facts=
        extra_facts,
)


# =========================================================
# GENERATE IMAGE PROMPT
# =========================================================

if st.button(
    "📝 Generate Image Prompt",
    use_container_width=True,
):

    try:

        image_prompt = build_image_prompt(

            profile,

            image_mode,
        )


        st.session_state[
            "image_prompt"
        ] = image_prompt


        st.success(
            "🎉 Image prompt generated!"
        )


    except Exception as error:

        st.error(
            f"Image prompt generation failed: {error}"
        )


# =========================================================
# SHOW IMAGE PROMPT
# =========================================================

if st.session_state.get(
    "image_prompt"
):

    st.subheader(
        "📋 Image Generation Prompt"
    )


    st.text_area(

        "Prompt",

        st.session_state[
            "image_prompt"
        ],

        height=500,

        key="image_prompt_output",
    )


    st.download_button(

        "⬇️ Download Prompt",

        st.session_state[
            "image_prompt"
        ],

        file_name=
            "etsy_image_prompt.txt",

        mime=
            "text/plain",
    )


# =========================================================
# GENERATE REAL IMAGES
# =========================================================

if st.button(

    "✨ Generate Images",

    type="primary",

    use_container_width=True,
):

    if not uploaded_files:

        st.error(
            "Please upload at least one product image first."
        )

        st.stop()


    if not os.getenv(
        "OPENAI_API_KEY"
    ):

        st.error(
            "OPENAI_API_KEY is not configured."
        )

        st.stop()


    try:

        image_prompt = build_image_prompt(

            profile,

            image_mode,
        )


        source_image = uploaded_files[0]


        with st.spinner(
            "🎨 Creating your Etsy product image..."
        ):

            generated_images = generate_images(

                image_bytes=
                    source_image.getvalue(),

                image_prompt=
                    image_prompt,

                mode=
                    image_mode,

                mime_type=
                    source_image.type
                    or "image/png",

                number_of_images=
                    number_of_images,
            )


        st.session_state[
            "generated_images"
        ] = generated_images


        st.session_state[
            "generated_image_mode"
        ] = image_mode


        st.success(

            f"🎉 Generated "
            f"{len(generated_images)} image(s)!"
        )


    except Exception as error:

        st.error(
            f"Image generation failed: {error}"
        )


# =========================================================
# DISPLAY GENERATED IMAGES
# =========================================================

generated_images = st.session_state.get(
    "generated_images"
)


if generated_images:

    st.divider()

    st.header(
        "🖼 Generated Images"
    )


    image_columns = st.columns(

        min(
            4,
            len(generated_images),
        )
    )


    for index, image_bytes in enumerate(
        generated_images
    ):

        with image_columns[
            index % len(image_columns)
        ]:

            st.image(

                image_bytes,

                caption=
                    f"{image_mode} #{index + 1}",

                use_container_width=True,
            )


            st.download_button(

                "⬇️ Download",

                data=
                    image_bytes,

                file_name=(

                    "etsy_"
                    +
                    image_mode.lower().replace(
                        " ",
                        "_",
                    )
                    +
                    f"_{index + 1}.png"
                ),

                mime=
                    "image/png",

                key=
                    f"download_image_{index}",
            )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Etsy AI — Product Analysis • Etsy SEO • "
    "Image Studio • AI Product Photography"
)
