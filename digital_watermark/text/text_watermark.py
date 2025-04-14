class TextWatmark:

    # Zero-width characters for embedding
    ZERO_WIDTH_CHARS = [
        "\u200b",  # Zero-width space
        "\u200c",  # Zero-width non-joiner
        "\u200d",  # Zero-width joiner
        "\u2060",  # Word joiner
        "\u2061",  # Function application
        "\u2062",  # Invisible times
        "\u2063",  # Invisible separator
        "\u2064",  # Invisible plus
        "\ufeff",  # Zero-width no-break space
    ]

    def __init__(self):
        pass

    def embed_watermark(self, text: str, watermark: str) -> str:
        """
        Embed the watermark into the text using zero-width characters.
        """
        pass

    def extract_watermark(self, text: str) -> str:
        """
        Extract the watermark from the text.
        """
        pass
