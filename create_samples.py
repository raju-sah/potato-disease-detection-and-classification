import math
import random
from PIL import Image, ImageDraw, ImageFilter

def create_healthy_leaf(path="static/samples/01_healthy_leaf.jpg"):
    img = Image.new("RGB", (512, 512), (238, 242, 238))
    draw = ImageDraw.Draw(img)
    
    # Background subtle gradient/texture
    for y in range(512):
        shade = int(235 + 15 * math.sin(y / 80.0))
        draw.line([(0, y), (511, y)], fill=(shade - 5, shade, shade - 5))
        
    # Draw leaf shape
    # Potato leaf is ovate with pointed tip
    leaf_color = (38, 145, 60)
    leaf_points = [
        (256, 40),   # tip
        (340, 110),
        (390, 200),
        (400, 300),
        (370, 390),
        (290, 440),
        (256, 470),  # base
        (222, 440),
        (142, 390),
        (112, 300),
        (122, 200),
        (172, 110),
    ]
    draw.polygon(leaf_points, fill=leaf_color)
    
    # Add botanical texture and shading
    overlay = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)
    
    # Inner healthy green gradients
    inner_points = [
        (256, 60),
        (325, 125),
        (365, 210),
        (375, 295),
        (345, 375),
        (280, 420),
        (256, 450),
        (232, 420),
        (167, 375),
        (137, 295),
        (147, 210),
        (187, 125),
    ]
    ov_draw.polygon(inner_points, fill=(52, 175, 75, 180))
    
    # Main midrib vein
    for i in range(50, 460):
        w = max(1, int(5 - (i - 50) * 3 / 410))
        ov_draw.line([(256, i), (256, i+1)], fill=(130, 210, 110, 240), width=w)
        
    # Secondary veins
    for vy in range(100, 420, 35):
        # Right vein
        vx_end = min(370, int(256 + 120 * math.sin((vy-60)/300.0 * math.pi)))
        vy_end = max(50, int(vy - 30))
        ov_draw.line([(256, vy), (vx_end, vy_end)], fill=(110, 195, 95, 180), width=2)
        # Left vein
        vx_end_l = max(140, int(256 - 120 * math.sin((vy-60)/300.0 * math.pi)))
        ov_draw.line([(256, vy), (vx_end_l, vy_end)], fill=(110, 195, 95, 180), width=2)
        
    img.paste(overlay, (0, 0), overlay)
    img = img.filter(ImageFilter.SMOOTH)
    img.save(path, quality=95)
    print(f"Created {path}")

def create_early_blight_leaf(path="static/samples/02_early_blight.jpg"):
    img = Image.new("RGB", (512, 512), (242, 240, 235))
    draw = ImageDraw.Draw(img)
    
    # Background
    for y in range(512):
        shade = int(238 + 10 * math.sin(y / 60.0))
        draw.line([(0, y), (511, y)], fill=(shade, shade - 2, shade - 8))
        
    leaf_points = [
        (256, 40),
        (340, 110),
        (390, 200),
        (400, 300),
        (370, 390),
        (290, 440),
        (256, 470),
        (222, 440),
        (142, 390),
        (112, 300),
        (122, 200),
        (172, 110),
    ]
    # Sickly yellowish green leaf
    draw.polygon(leaf_points, fill=(65, 130, 45))
    
    overlay = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)
    
    # Veins
    for i in range(50, 460):
        ov_draw.line([(256, i), (256, i+1)], fill=(115, 175, 80, 200), width=3)
        
    # Concentric target spots (Early Blight characteristic rings)
    spots = [
        (220, 220, 45),
        (320, 180, 35),
        (290, 320, 50),
        (180, 340, 38),
        (330, 280, 28)
    ]
    
    for cx, cy, rad in spots:
        # Yellow chlorotic halo
        ov_draw.ellipse([cx - rad - 12, cy - rad - 12, cx + rad + 12, cy + rad + 12], fill=(215, 190, 40, 160))
        # Concentric brown rings
        for r, col in [(rad, (110, 55, 20, 220)), 
                       (int(rad*0.8), (140, 75, 30, 220)), 
                       (int(rad*0.6), (90, 40, 15, 240)), 
                       (int(rad*0.4), (130, 70, 25, 240)), 
                       (int(rad*0.2), (50, 20, 5, 255))]:
            ov_draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)
            
    img.paste(overlay, (0, 0), overlay)
    img = img.filter(ImageFilter.SMOOTH)
    img.save(path, quality=95)
    print(f"Created {path}")

def create_late_blight_leaf(path="static/samples/03_late_blight.jpg"):
    img = Image.new("RGB", (512, 512), (235, 235, 235))
    draw = ImageDraw.Draw(img)
    
    for y in range(512):
        shade = int(230 + 12 * math.cos(y / 70.0))
        draw.line([(0, y), (511, y)], fill=(shade, shade, shade))
        
    leaf_points = [
        (256, 40),
        (340, 110),
        (390, 200),
        (400, 300),
        (370, 390),
        (290, 440),
        (256, 470),
        (222, 440),
        (142, 390),
        (112, 300),
        (122, 200),
        (172, 110),
    ]
    draw.polygon(leaf_points, fill=(45, 100, 40))
    
    overlay = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)
    
    # Veins
    for i in range(50, 460):
        ov_draw.line([(256, i), (256, i+1)], fill=(90, 140, 70, 180), width=3)
        
    # Large water-soaked black necrotic blotches (Late Blight)
    lesions = [
        # (center_x, center_y, rx, ry)
        (256, 120, 80, 60),
        (340, 260, 75, 90),
        (180, 270, 85, 75),
        (270, 380, 95, 65)
    ]
    
    for cx, cy, rx, ry in lesions:
        # Water-soaked pale margin
        ov_draw.ellipse([cx - rx - 10, cy - ry - 10, cx + rx + 10, cy + ry + 10], fill=(120, 130, 80, 140))
        # Black/dark brown necrosis
        ov_draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=(25, 20, 15, 230))
        # Inner rotting core
        ov_draw.ellipse([cx - int(rx*0.6), cy - int(ry*0.6), cx + int(rx*0.6), cy + int(ry*0.6)], fill=(10, 8, 8, 255))
        # White moldy sporulation edge
        for angle in range(0, 360, 20):
            rad_a = math.radians(angle)
            mx = cx + (rx - 4) * math.cos(rad_a) + random.uniform(-2, 2)
            my = cy + (ry - 4) * math.sin(rad_a) + random.uniform(-2, 2)
            ov_draw.ellipse([mx-4, my-4, mx+4, my+4], fill=(225, 230, 225, 170))
            
    img.paste(overlay, (0, 0), overlay)
    img = img.filter(ImageFilter.SMOOTH)
    img.save(path, quality=95)
    print(f"Created {path}")

if __name__ == "__main__":
    create_healthy_leaf()
    create_early_blight_leaf()
    create_late_blight_leaf()
