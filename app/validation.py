"""Upload validation shared by the web app and the model API (Task 4.2)."""
import io
import os

from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
ALLOWED_FORMATS = {"JPEG", "PNG"}
MAX_PIXELS = 50_000_000


class ValidationError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.message, self.status = message, status


def validate_upload(file_storage):
    """Return (clean_filename, image_bytes). Nothing is written to disk."""
    if file_storage is None or not file_storage.filename:
        raise ValidationError("No file was selected. Choose a JPG or PNG image and try again.")
    filename = secure_filename(os.path.basename(file_storage.filename)) or "upload"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise ValidationError(f"'{filename}' is not a supported type. Upload a .jpg, .jpeg or .png file.")
    data = file_storage.read()
    if not data:
        raise ValidationError("The file is empty.")
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt, size = im.format, im.size
            im.verify()
        if fmt not in ALLOWED_FORMATS:
            raise ValidationError(f"The file content is {fmt}, not a JPEG or PNG image.")
        if size[0] * size[1] > MAX_PIXELS:
            raise ValidationError("The image has too many pixels. Use an image under 50 megapixels.")
        with Image.open(io.BytesIO(data)) as im:
            im.load()   # full decode: catches truncated / corrupted files
    except ValidationError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError):
        raise ValidationError(
            f"'{filename}' has an image extension but its contents cannot be decoded as an image.")
    return filename, data
