import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

from transformers import MarianMTModel, MarianTokenizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('image_caption_model.pth', map_location=device)

# Load pre-trained BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image_path):
    # Load and preprocess the image
    raw_image = Image.open(image_path).convert("RGB")
    
    # Preprocess the image and generate the caption
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs)
    
    # Decode the generated caption
    caption = processor.decode(out[0], skip_special_tokens=True)
    
    return caption




class Translator:
    def __init__(self, source_lang="en", target_lang="es"):  # Updated target_lang to Malayalam
        """
        Initialize the translation model.
        :param source_lang: Source language code (default: "en" for English).
        :param target_lang: Target language code (e.g., "es" for spanish).
        """
        model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)

    def translate(self, text):
        """
        Translate the input text to the target language.
        :param text: Input text in the source language.
        :return: Translated text in the target language.
        """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        translated = self.model.generate(**inputs)
        return self.tokenizer.decode(translated[0], skip_special_tokens=True)

def Translate_caption(caption):
    source_language = "en"
    target_language = "es"  # Translate to spanish
    translator = Translator(source_lang=source_language, target_lang=target_language)
    translated_caption = translator.translate(caption)
    print("\nTranslated Caption (Spanish):")
    print(translated_caption)
    return translated_caption


