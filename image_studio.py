"""
Etsy AI V2 - Image Studio

Creates structured prompts for:
1. Product Only Hero
2. Background Replacement
3. Lifestyle Scene
4. Model Wearing
"""

from dataclasses import dataclass


@dataclass
class ProductProfile:
    jewelry_type: str = ""
    material: str = ""
    gemstone: str = ""
    dimensions: str = ""
    style: str = ""
    extra_facts: str = ""


BASE_PRESERVATION_RULES = """
Preserve the supplied jewelry exactly as the source product.

Do not redesign, recolor, reshape, simplify, enlarge, shrink disproportionately,
add, remove, or substitute any product component.

Preserve:
- metal finish
- gemstone appearance
- gemstone color
- setting
- engraving
- texture
- chain
- clasp
- thickness
- proportions
- construction
- surface details

The generated image must look like a professional photograph
of the same physical jewelry.

Never invent product specifications.
""".strip()


SCENE_PRESETS = {

    "Product Only Hero": """
Create a premium Etsy product-only hero photograph.

No people.
No hands.
No model.

Place the jewelry naturally on an elegant surface such as:
walnut wood, black leather, travertine, slate, marble,
or another understated luxury material.

Keep the jewelry as the visual center.

Use:
- clean negative space
- excellent Etsy thumbnail readability
- realistic contact shadows
- realistic environmental reflections
- soft depth of field
- premium jewelry photography
- refined masculine or feminine styling according to the product

Do not allow props to compete with the jewelry.
""".strip(),


    "Background Replacement": """
Replace only the original background.

Keep the jewelry itself completely unchanged.

Create a believable physical environment around the jewelry.

Use:
- realistic contact shadow
- environmental reflections
- natural light bounce
- accurate metal reflection
- realistic gemstone response
- subtle depth of field

The jewelry must look physically placed in the new environment,
not pasted onto it.

No visible cutout edges.
""".strip(),


    "Lifestyle Scene": """
Create a sophisticated lifestyle photograph appropriate for Etsy.

Choose an environment that naturally complements the jewelry's:
material, gemstone, style, color, and target customer.

The jewelry remains the main visual focus.

Use realistic:
- scale
- perspective
- shadows
- reflections
- depth of field
- environmental lighting

Keep props minimal and tasteful.
""".strip(),


    "Model Wearing": """
Create a polished jewelry lifestyle photograph featuring
a model appropriate for the product's target customer.

The model's:
- clothing
- hairstyle
- makeup
- environment
- overall styling

should complement the jewelry.

Frame the photograph so the jewelry is clearly visible.

The jewelry itself must remain exactly unchanged.

Use realistic skin, fabric, reflections, shadows,
perspective, and natural photographic lighting.
""".strip(),
}


def choose_scene_style(profile: ProductProfile) -> str:

    text = " ".join([
        profile.jewelry_type,
        profile.material,
        profile.gemstone,
        profile.style,
        profile.extra_facts,
    ]).lower()


    if any(word in text for word in [
        "dark nature",
        "gothic",
        "oxidized",
        "obsidian",
    ]):

        return (
            "dark botanical luxury, natural stone, "
            "dark wood, restrained cinematic lighting"
        )


    if any(word in text for word in [
        "jade",
        "jadeite",
        "nephrite",
        "pearl",
        "mother of pearl",
    ]):

        return (
            "quiet luxury, warm neutral stone, "
            "soft botanical accents, elegant natural daylight"
        )


    if any(word in text for word in [
        "men",
        "men's",
        "mens",
        "dog tag",
    ]):

        return (
            "masculine premium lifestyle, walnut, "
            "black leather, stone, understated architecture"
        )


    if any(word in text for word in [
        "vintage",
        "retro",
        "antique",
    ]):

        return (
            "refined vintage editorial styling, aged wood, "
            "linen, stone, soft directional light"
        )


    return (
        "modern quiet luxury, neutral stone, refined wood, "
        "clean editorial jewelry photography"
    )


def build_image_prompt(
    profile: ProductProfile,
    mode: str,
) -> str:

    if mode not in SCENE_PRESETS:
        raise ValueError(
            f"Unsupported image mode: {mode}"
        )


    scene_style = choose_scene_style(profile)


    product_facts = f"""
Confirmed product facts:

Jewelry type:
{profile.jewelry_type}

Material:
{profile.material}

Gemstone:
{profile.gemstone}

Dimensions:
{profile.dimensions}

Style:
{profile.style}

Other confirmed facts:
{profile.extra_facts}
""".strip()


    return f"""
{BASE_PRESERVATION_RULES}


{product_facts}


IMAGE MODE:

{mode}


SCENE DIRECTION:

{SCENE_PRESETS[mode]}


RECOMMENDED VISUAL STYLE:

{scene_style}


LIGHTING:

Use physically believable global illumination.

Use a warm key light and subtle cool fill/rim light
where appropriate.

Integrate:

- environmental reflections
- natural light bounce
- realistic contact shadows
- accurate metal reflections
- realistic gemstone refraction
- believable material micro-texture

Avoid:
- artificial halos
- hard cutout edges
- plastic-looking metal
- excessive glow
- overprocessed HDR
- fake reflections


COMPOSITION:

Square 1:1 composition.

Optimize for Etsy product-thumbnail readability.

The jewelry should occupy approximately 60–80%
of the meaningful visual area while maintaining elegant
negative space.

The product must remain the primary visual subject.


NEGATIVE CONSTRAINTS:

No redesign.
No duplicate jewelry.
No extra jewelry.
No extra stones.
No changed gemstone color.
No altered chain.
No altered clasp.
No altered engraving.
No distorted proportions.
No floating product.
No pasted cutout appearance.
No excessive props.
No text.
No watermark.
No logo.
No fake certification.
No unsupported product claims.
""".strip()


def get_scene_options() -> list[str]:

    return list(
        SCENE_PRESETS.keys()
    )
