import streamlit as st
from openai import OpenAI
from image_studio import from image_generator import generate_images (
    ProductProfile,
    build_image_prompt,
    get_scene_options,
)
import base64
import json
import os

import streamlit as st
from openai import OpenAI


# =========================================================
# Page setup
# =========================================================

st.set_page_config(
    page_title="Etsy AI",
    page_icon="💎",
    layout="wide",
)

st.title("💎 Etsy AI")
st.subheader("AI Jewelry Listing & Image Studio")

st.write(
    "Upload your jewelry images, provide confirmed product facts, "
    "and generate Etsy-ready SEO content."
)


# =========================================================
# AI System Prompt
# =========================================================

SYSTEM_PROMPT = """
You are an expert Etsy SEO strategist, jewelry product analyst,
and American e-commerce copywriter.

Your target market is primarily US Etsy shoppers.

Analyze the supplied jewelry images carefully.

IMPORTANT FACTUAL RULE:
Never invent product facts.

Do NOT assume:
- 925 sterling silver
- gold plating
- gemstone identity
- natural stone
- dimensions
- weight
- certification
- handmade status
- healing properties
- origin
- durability claims

unless the seller explicitly provides or confirms them.

If something cannot be reliably determined from the image,
describe it visually or leave it out.

Use natural American English.

Avoid keyword stuffing.

The Etsy title should be readable, shopper-friendly,
and SEO-aware.

Generate exactly 13 Etsy tags.
Each tag must contain no more than 20 characters.

Return ONLY valid JSON.

Required JSON structure:

{
  "product_analysis": {
    "jewelry_type": "",
    "materials": [],
    "gemstone": "",
    "colors": [],
    "design_details": [],
    "style": "",
    "visual_positioning": ""
  },
  "seo_title": "",
  "product_description": "",
  "etsy_tags": [],
  "search_keywords": [],
  "buyer_benefits": [],
  "gift_occasions": []
}
"""


# =========================================================
# Sidebar - confirmed product information
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
        placeholder=(
            "Add any information you want the AI to use.\n"
            "Example: gold plated, handmade, natural stone, etc."
        ),
    )


# =========================================================
# Image upload
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


# =========================================================
# Display uploaded images
# =========================================================

if uploaded_files:

    st.success(
        f"{len(uploaded_files)} product image(s) uploaded."
    )

    columns = st.columns(
        min(4, len(uploaded_files))
    )

    for index, uploaded_file in enumerate(uploaded_files):

        with columns[index % len(columns)]:

            st.image(
                uploaded_file,
                caption=uploaded_file.name,
                use_container_width=True,
            )


# =========================================================
# Generate Listing
# =========================================================

if uploaded_files:

    st.header("✨ 2. Generate Etsy Listing")

    generate_button = st.button(
        "🚀 Analyze Product & Generate Etsy Listing",
        type="primary",
        use_container_width=True,
    )

    if generate_button:

        if not os.getenv("OPENAI_API_KEY"):

            st.error(
                "OPENAI_API_KEY is not configured. "
                "Please add your OpenAI API key to the app environment."
            )

            st.stop()


        client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"]
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
                "text": confirmed_information
                + """

Please analyze the uploaded product images.

First identify the jewelry visually.

Then combine the visual information
with the confirmed seller information.

Generate:

1. Product Analysis
2. Etsy SEO Title
3. Product Description
4. Exactly 13 Etsy Tags
5. Search Keywords
6. Buyer Benefits
7. Gift Occasions

The product itself must remain the source of truth.
Do not add unsupported claims.
""",
            }
        ]


        # -------------------------------------------------
        # Add product images
        # -------------------------------------------------

        for uploaded_file in uploaded_files:

            image_bytes = uploaded_file.getvalue()

            encoded_image = base64.b64encode(
                image_bytes
            ).decode("utf-8")

            mime_type = (
                uploaded_file.type
                or "image/jpeg"
            )

            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": (
                            f"data:{mime_type};"
                            f"base64,{encoded_image}"
                        )
                    },
                }
            )


        # -------------------------------------------------
        # Call AI
        # -------------------------------------------------

        with st.spinner(
            "🔍 Analyzing jewelry and generating Etsy SEO..."
        ):

            try:

                response = client.chat.completions.create(

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


                raw_result = (
                    response
                    .choices[0]
                    .message
                    .content
                )


                result = json.loads(
                    raw_result
                )


                # Store result
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
# Display Results
# =========================================================

result = st.session_state.get(
    "etsy_listing"
)


if result:

    st.divider()

    st.header(
        "🛍️ Etsy Listing Results"
    )


    # =====================================================
    # Product Analysis
    # =====================================================

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
                ""
            )
        )


        st.write(
            "**Material**"
        )

        st.write(
            analysis.get(
                "materials",
                []
            )
        )


        st.write(
            "**Gemstone**"
        )

        st.write(
            analysis.get(
                "gemstone",
                ""
            )
        )


        st.write(
            "**Style**"
        )

        st.write(
            analysis.get(
                "style",
                ""
            )
        )


    with col2:

        st.write(
            "**Colors**"
        )

        st.write(
            analysis.get(
                "colors",
                []
            )
        )


        st.write(
            "**Design Details**"
        )

        for detail in analysis.get(
            "design_details",
            []
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
                ""
            )
        )


    # =====================================================
    # SEO TITLE
    # =====================================================

    st.divider()

    st.subheader(
        "🏷️ Etsy SEO Title"
    )


    title = result.get(
        "seo_title",
        ""
    )


    st.text_area(
        "Title",
        title,
        height=100,
        key="seo_title_output",
    )


    # =====================================================
    # DESCRIPTION
    # =====================================================

    st.subheader(
        "📝 Product Description"
    )


    description = result.get(
        "product_description",
        ""
    )


    st.text_area(
        "Description",
        description,
        height=350,
        key="description_output",
    )


    # =====================================================
    # TAGS
    # =====================================================

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
            "Etsy listing should contain exactly 13 tags."
        )


    tag_columns = st.columns(3)


    for index, tag in enumerate(tags):

        with tag_columns[index % 3]:

            st.code(
                tag,
                language=None,
            )


    # =====================================================
    # SEARCH KEYWORDS
    # =====================================================

    st.subheader(
        "🔎 Search Keywords"
    )


    keywords = result.get(
        "search_keywords",
        []
    )


    for keyword in keywords:

        st.write(
            f"• {keyword}"
        )


    # =====================================================
    # BUYER BENEFITS
    # =====================================================

    st.subheader(
        "💎 Buyer Benefits"
    )


    benefits = result.get(
        "buyer_benefits",
        []
    )


    for benefit in benefits:

        st.write(
            f"• {benefit}"
        )


    # =====================================================
    # GIFT OCCASIONS
    # =====================================================

    st.subheader(
        "🎁 Gift Occasions"
    )


    occasions = result.get(
        "gift_occasions",
        []
    )


    for occasion in occasions:

        st.write(
            f"• {occasion}"
        )


    # =====================================================
    # Copy-ready JSON
    # =====================================================

    st.divider()

    st.subheader(
        "📦 Complete Listing Data"
    )


    st.json(
        result
    )


    st.info(
        "V1 complete. Next: AI background replacement, "
        "lifestyle scenes, model-wearing images, "
        "and product-only Etsy hero images."
    )
# =========================================================
# V2 - AI IMAGE STUDIO
# =========================================================

st.divider()

st.header("🖼️ AI Image Studio")

st.write(
    "Create professional image-generation prompts while "
    "preserving the original jewelry design."
)


image_mode = st.selectbox(
    "Choose Image Mode",
    get_scene_options(),
)


st.subheader("🎨 Scene Settings")

scene_col1, scene_col2 = st.columns(2)


with scene_col1:

    st.write("**Selected Mode**")

    st.info(image_mode)


with scene_col2:

    st.write("**Recommended Style**")

    st.write(
        "Automatically selected according to "
        "the product's material, gemstone, and style."
    )


if st.button(
    "✨ Generate Image Prompt",
    type="primary",
    use_container_width=True,
):

    profile = ProductProfile(

        jewelry_type=analysis.get(
            "jewelry_type",
            product_name,
        ),

        material=(
            material
            or ", ".join(
                analysis.get(
                    "materials",
                    []
                )
            )
        ),

        gemstone=(
            gemstone
            or analysis.get(
                "gemstone",
                ""
            )
        ),

        dimensions=dimensions,

        style=(
            style
            or analysis.get(
                "style",
                ""
            )
        ),

        extra_facts=extra_facts,
    )


    try:

        image_prompt = build_image_prompt(
            profile,
            image_mode,
        )


        st.success(
            "🎉 Image prompt generated!"
        )


        st.subheader(
            "📋 Image Generation Prompt"
        )


        st.text_area(
            "Copy this prompt into your image-generation workflow",
            image_prompt,
            height=700,
            key="image_prompt_output",
        )


        st.download_button(
            "⬇️ Download Prompt",
            image_prompt,
            file_name="etsy_image_prompt.txt",
            mime="text/plain",
        )


    except Exception as error:

        st.error(
            f"Image prompt generation failed: {error}"
        )
