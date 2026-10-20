import random
from PIL import Image, ImageDraw, ImageFont


def generate_captcha(
    image_size=(260, 80),
    num_digits=5,
    font_size_range=(30, 40),
    minimum_padding=(5, 10),  # 给边缘留白避免数字被截断
    rotation_angle_range=(-30, 30),
    max_noisy_lines=5,
    noise_ratio=0.02,
    border_width=2,
    seed=None,
):
    width, height = image_size
    pad_x, pad_y = minimum_padding

    rng = random.Random(seed)
    digits = [str(rng.randint(0, 9)) for _ in range(num_digits)]  # 随机生成5个数字
    label = "".join(digits)

    image = Image.new("RGB", (width, height), (255, 255, 255))
    slot_width = (width - 2 * pad_x) / num_digits  # 每个数字的槽宽度，保证数字之间不会重叠

    for i, digit in enumerate(digits):

        # 确定数字大小
        font_size = rng.randint(font_size_range[0], font_size_range[1])
        font = ImageFont.load_default(size=font_size)

        # 确定数字颜色
        color = (rng.randint(0, 160), rng.randint(0, 160), rng.randint(0, 160))

        # 画出数字
        left, top, right, bottom = ImageDraw.Draw(image).textbbox((0, 0), digit, font=font)
        digit_img = Image.new("RGBA", (right - left, bottom - top), (0, 0, 0, 0))
        ImageDraw.Draw(digit_img).text((-left, -top), digit, font=font, fill=color + (255,))

        # 旋转数字
        if rotation_angle_range is not None:
            angle = rng.uniform(rotation_angle_range[0], rotation_angle_range[1])
            digit_img = digit_img.rotate(angle, expand=True)
        w, h = digit_img.size

        # 在槽内随机位置放置数字
        slot_left = pad_x + i * slot_width
        x_min = int(slot_left)
        x_max = int(slot_left + slot_width - w)
        y_min = pad_y
        y_max = height - pad_y - h
        x = rng.randint(x_min, max(x_min, x_max))
        y = rng.randint(y_min, max(y_min, y_max))

        # 将数字贴到背景图上
        image.paste(digit_img, (x, y), digit_img)

    # 画干扰线
    draw = ImageDraw.Draw(image)
    for _ in range(rng.randint(0, max_noisy_lines)):
        # 确定干扰线端点和颜色
        start = (rng.randint(0, width - 1), rng.randint(0, height - 1))
        end = (rng.randint(0, width - 1), rng.randint(0, height - 1))
        line_color = (rng.randint(0, 200), rng.randint(0, 200), rng.randint(0, 200))
        draw.line([start, end], fill=line_color, width=rng.randint(1, 2))

    # 画噪点
    pixels = image.load()
    for _ in range(int(width * height * noise_ratio)):
        px, py = rng.randint(0, width - 1), rng.randint(0, height - 1)
        pixels[px, py] = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))

    # 在边缘画边框
    if border_width > 0:
        draw.rectangle([0, 0, width - 1, height - 1], outline=(0, 0, 0), width=border_width)

    return image, label
