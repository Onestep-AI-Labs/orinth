from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from app.schemas import Detection


CLASS_COLORS = {
    "granuloma": (70, 180, 160),
    "kista": (250, 128, 114),
    "Normal": (116, 116, 116),
}


def create_overlay(image_path: Path, detections: list[Detection], output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()

    for detection in detections:
        color = CLASS_COLORS.get(detection.class_name, (120, 120, 120))
        rgba_fill = (*color, 90)
        rgba_line = (*color, 230)
        points = [(float(x), float(y)) for x, y in detection.polygon]
        if len(points) >= 3:
            draw.polygon(points, fill=rgba_fill, outline=rgba_line)
        box = detection.bbox
        xyxy = [box.x, box.y, box.x + box.width, box.y + box.height]
        draw.rectangle(xyxy, outline=rgba_line, width=3)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        text_position = (box.x, max(0, box.y - 14))
        draw.rectangle(
            [text_position, (text_position[0] + max(76, len(label) * 6), text_position[1] + 13)],
            fill=(*color, 220),
        )
        draw.text((text_position[0] + 3, text_position[1] + 1), label, fill=(255, 255, 255), font=font)

    composited = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    composited.save(output_path, quality=92)
