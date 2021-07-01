
import logging
import os
import random
import shutil

# It is a convenient method for simple scripts to configure the log package at one time.
logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)


def main() -> None:
    r""" Make training data set and validation data set."""
    total_image_lists = os.listdir(os.path.join("train", "input"))
    # The original data set is divided into 9:1 (train:test)
    test_image_lists = random.sample(total_image_lists, int(len(total_image_lists) / 10))
    # Move the validation set to the specified location.
    for test_image_name in test_image_lists:
        logger.info(f"Process: `{os.path.join('train', 'input', test_image_name)}`.")
        # Move the test image into the test folder.
        shutil.move(os.path.join("train", "input", test_image_name), os.path.join("test", "input", test_image_name))
        shutil.move(os.path.join("train", "target", test_image_name), os.path.join("test", "target", test_image_name))


if __name__ == "__main__":
    logger.info("ScriptEngine:")
    logger.info("\tAPI version .......... 0.3.0")
    

    main()
