#数据预处理，将图片分块
import logging
import os

from PIL import Image

# It is a convenient method for simple scripts to configure the log package at one time.
logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)


def main(dir_path: str = None) -> None:
    # Get all the images under the folder.
    for file in os.listdir(dir_path):
        logger.info(f"Process: `{os.path.join(dir_path, file)}`.")
        # Get all cut image data.
        crop_images = crop_image(Image.open(os.path.join(dir_path, file)))
        # Save all captured image data in turn.
        save_images(os.path.join(dir_path, file), crop_images)
        # Delete original image.
        os.remove(os.path.join(dir_path, file))


def crop_image(image) -> list:
    assert image.size[0] != image.size[1]
    # Get the split image size.
    crop_image_size = int(image.size[0] / 9)
    # (left, upper, right, lower)
    box_list = []
    for width_index in range(0, 9):
        for height_index in range(0, 9):
            box = (height_index * crop_image_size,
                   width_index * crop_image_size,
                   (height_index + 1) * crop_image_size,
                   (width_index + 1) * crop_image_size)

            box_list.append(box)

    # Save all split image data.
    crop_images = [image.crop(box) for box in box_list]
    return crop_images


def save_images(raw_filename, image_list) -> None:
    index = 0
    for image in image_list:
        image.save(raw_filename.split(".")[0] + "_" + str(index) + ".png")
        index += 1


if __name__ == "__main__":
    logger.info("ScriptEngine:")
    logger.info("\tAPI version .......... 0.3.0")
    logger.info("\tBuild ................ 2021.06.13")

    main()
