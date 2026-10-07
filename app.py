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

from relief_generator import ReliefConfig, generate_relief_stl, preview_heightmap, preview_hillshade, export_relief_glb, inspect_stl, validate_relief_config


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
# V3 JEWELRY MEMORIAL PORTRAIT COIN
# =========================================================

st.divider()
st.header("🪙 3. V4 Jewelry Memorial Portrait Coin")
st.write(
    "Turn a portrait into a jewelry-style memorial coin relief with optional "
    "face focus, raised border rings, personalized text, hanging hole, and STL inspection."
)

relief_file = st.file_uploader(
    "Upload portrait / memorial photo / artwork",
    type=["jpg", "jpeg", "png", "webp"],
    key="relief_upload",
)

if relief_file:
    from memorial_coin_v3 import (
        MemorialCoinConfig,
        generate_memorial_coin_stl,
        make_memorial_coin_depth,
        preview_memorial_coin,
        inspect_memorial_coin,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        coin_style = st.selectbox(
            "Coin Relief Style",
            ["Classic Coin", "Deep Relief", "Soft Relief"],
            key="v3_coin_style",
        )
        relief_shape = st.selectbox(
            "Pendant Shape",
            ["Circle", "Oval", "Heart", "Dog Tag"],
            key="v3_relief_shape",
        )
        relief_depth_mode = st.selectbox(
            "Depth Method",
            ["AI Depth", "Grayscale"],
            key="v3_relief_depth_mode",
        )
    with col2:
        relief_width = st.number_input(
            "Width (mm)", 20.0, 100.0, 30.0, 1.0, key="v3_relief_width"
        )
        relief_height = st.number_input(
            "Height (mm)", 20.0, 100.0, 30.0, 1.0, key="v3_relief_height"
        )
        relief_base = st.number_input(
            "Base Thickness (mm)", 1.2, 8.0, 2.2, 0.1, key="v3_relief_base"
        )
    with col3:
        relief_z = st.number_input(
            "Portrait Relief Height (mm)", 0.4, 4.0, 1.1, 0.1, key="v3_relief_z"
        )
        relief_hole = st.number_input(
            "Hanging Hole (mm)", 2.0, 8.0, 3.0, 0.1, key="v3_relief_hole"
        )
        relief_resolution = st.select_slider(
            "Mesh Resolution", [160, 200, 240, 300], value=200, key="v3_relief_resolution"
        )

    face_focus = st.checkbox(
        "🎯 Face focus / facial detail enhancement",
        value=True,
        key="v3_face_focus",
    )

    st.subheader("🧑‍🎨 V4 Portrait Sculpting")
    sculpt_col1, sculpt_col2, sculpt_col3 = st.columns(3)
    with sculpt_col1:
        feature_protection = st.slider("Face Feature Protection", 0.0, 1.0, 0.72, 0.05, key="v4_feature_protection")
    with sculpt_col2:
        hair_preservation = st.slider("Hair / Silhouette Preservation", 0.0, 1.0, 0.45, 0.05, key="v4_hair_preservation")
    with sculpt_col3:
        background_flatten = st.slider("Background Flattening", 0.0, 1.0, 0.78, 0.05, key="v4_background_flatten")

    st.subheader("🏭 V5 Production Mode")
    from memorial_coin_v3 import get_production_preset, validate_memorial_coin_production

    prod_col1, prod_col2 = st.columns(2)
    with prod_col1:
        production_size = st.selectbox(
            "Jewelry Size Preset",
            ["25mm Coin", "30mm Coin", "35mm Coin"],
            index=1,
            key="v5_production_size",
        )
    with prod_col2:
        production_process = st.selectbox(
            "Manufacturing Process",
            ["Jewelry Casting", "Resin / 3D Print", "CNC / Engraving"],
            key="v5_production_process",
        )

    production_preset = get_production_preset(production_size, production_process)
    st.caption(
        f"Auto guideline: base {production_preset['base_mm']:.2f} mm · "
        f"relief {production_preset['relief_mm']:.2f} mm · "
        f"min detail {production_preset['min_detail_mm']:.2f} mm · "
        f"min text line {production_preset['min_text_line_mm']:.2f} mm"
    )

    auto_apply = st.checkbox(
        "⚙️ Apply V5 production preset to base / relief / hole / border",
        value=True,
        key="v5_auto_apply",
    )
    if auto_apply:
        relief_base = production_preset["base_mm"]
        relief_z = production_preset["relief_mm"]
        relief_hole = production_preset["hole_mm"]
        border_width = production_preset["border_mm"]
        border_height = production_preset["border_height_mm"]

    st.subheader("⭕ Jewelry Coin Structure")
    ring_col1, ring_col2, ring_col3 = st.columns(3)
    with ring_col1:
        border_width = st.slider("Outer Border Width (mm)", 0.5, 3.0, 1.2, 0.1, key="v3_border_width")
    with ring_col2:
        inner_ring = st.slider("Inner Ring Width (mm)", 0.2, 1.5, 0.55, 0.05, key="v3_inner_ring")
    with ring_col3:
        border_height = st.slider("Border Height (mm)", 0.1, 0.6, 0.25, 0.05, key="v3_border_height")

    st.subheader("✍️ Personalization")
    text_col1, text_col2, text_col3 = st.columns(3)
    with text_col1:
        memorial_text = st.text_input(
            "Memorial Text",
            placeholder="Example: Forever Loved • 1982–2026",
            max_chars=36,
            key="v3_memorial_text",
        )
    with text_col2:
        text_mode = st.selectbox("Text Style", ["Raised", "Engraved"], key="v3_text_mode")
    with text_col3:
        text_position = st.selectbox("Text Position", ["Bottom", "Top", "Center"], key="v3_text_position")

    cfg = MemorialCoinConfig(
        relief=ReliefConfig(
            width_mm=relief_width,
            height_mm=relief_height,
            base_thickness_mm=relief_base,
            relief_height_mm=relief_z,
            hole_diameter_mm=relief_hole,
            resolution=relief_resolution,
            shape=relief_shape,
            depth_model=relief_depth_mode,
            relief_mode="Photo",
            invert_depth=False,
            depth_gamma=0.85,
        ),
        coin_style=coin_style,
        face_focus=face_focus,
        border_width_mm=border_width,
        border_height_mm=border_height,
        inner_ring_width_mm=inner_ring,
        text=memorial_text,
        text_height_mm=0.25,
        text_mode=text_mode,
        text_position=text_position,
        feature_protection=feature_protection,
        hair_preservation=hair_preservation,
        background_flatten=background_flatten,
    )

    if st.button("🔎 Preview V4 Portrait Coin", use_container_width=True, key="v3_preview"):
        try:
            with st.spinner("Analyzing portrait and building coin relief preview..."):
                preview = preview_memorial_coin(relief_file.getvalue(), cfg)
            st.image(preview, caption="V4 Jewelry Relief Height Map", use_container_width=True)
        except Exception as error:
            st.error(f"V5 preview failed: {error}")

    if st.button(
        "🪙 Generate V4 Memorial Coin STL",
        type="primary",
        use_container_width=True,
        key="v3_generate_stl",
    ):
        try:
            with st.spinner("Building jewelry memorial coin STL..."):
                stl_bytes = generate_memorial_coin_stl(relief_file.getvalue(), cfg)
            inspection = inspect_memorial_coin(stl_bytes)
            st.session_state["v4_stl"] = stl_bytes
            st.session_state["v4_inspection"] = inspection
            st.success("V4 memorial coin STL generated.")
        except Exception as error:
            st.error(f"V5 STL generation failed: {error}")

    inspection = st.session_state.get("v4_inspection")
    if inspection:
        st.subheader("📐 Production Check")
        st.json(inspection)
        if inspection.get("watertight"):
            st.success("Mesh reports watertight.")
        else:
            st.error("Mesh is not watertight. Do not send this STL to production yet.")

    if st.session_state.get("v4_stl"):
        st.download_button(
            "⬇️ Download V5 STL",
            st.session_state["v4_stl"],
            "jewelry_memorial_portrait_coin_v5.stl",
            "model/stl",
            use_container_width=True,
            key="v3_download_stl",
        )

    st.info(
        "V5 adds production presets, process-aware guidelines, and manufacturability checks. V4 uses optional OpenCV face localization when available. Face feature protection, hair/silhouette preservation, and background flattening are shaping aids rather than biometric identification. If face detection "
        "is unavailable, it safely falls back to the V2-style depth workflow."
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
