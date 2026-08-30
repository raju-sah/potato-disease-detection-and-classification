import nbformat
import json

with open("kaggle/potato-leaf-disease-classification-efficientnetb3.ipynb", "r") as f:
    nb = nbformat.read(f, as_version=4)

# Replace EfficientNetB3 with EfficientNetV2B3
for cell in nb.cells:
    if cell.cell_type == "code" or cell.cell_type == "markdown":
        # Rename everywhere
        cell.source = cell.source.replace("EfficientNetB3", "EfficientNetV2B3")
        cell.source = cell.source.replace("base_b3", "base_v2")
        cell.source = cell.source.replace("model_b3", "model_v2")

        # Update ImageDataGenerator
        if "ImageDataGenerator(" in cell.source and "rotation_range=" in cell.source:
            cell.source = cell.source.replace(
                "rotation_range=25,",
                "rotation_range=30,\n    width_shift_range=0.2,\n    height_shift_range=0.2,\n    shear_range=0.15,\n    zoom_range=0.25,\n    horizontal_flip=True,\n    vertical_flip=True,\n    brightness_range=[0.7, 1.4],\n    channel_shift_range=25.0,"
            )
            # Remove the old duplicates
            lines = cell.source.split('\n')
            new_lines = []
            skip = False
            for line in lines:
                if "width_shift_range=" in line and "0.2" not in line: continue
                if "height_shift_range=" in line and "0.2" not in line: continue
                if "shear_range=" in line and "0.15" not in line: continue
                if "zoom_range=" in line and "0.25" not in line: continue
                if "horizontal_flip=" in line and "True" not in line: continue
                if "vertical_flip=" in line and "True" not in line: continue
                if "brightness_range=" in line and "0.7" not in line: continue
                if "channel_shift_range=" in line and "25.0" not in line: continue
                new_lines.append(line)
            # just re-injecting a cleaner version
            
# Let's do it cleanly by searching and replacing exact blocks.
