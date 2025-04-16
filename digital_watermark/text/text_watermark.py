import random
import hashlib
from typing import List, Tuple
import re


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

    HOMOGLYPHS = {
        "a": ["а", "ａ", "𝐚", "𝑎"],
        "b": ["ｂ", "𝐛", "𝑏", "𝒃"],
        "c": ["с", "ｃ", "𝐜", "𝑐"],
        "d": ["ԁ", "ｄ", "𝐝", "𝑑"],
        "e": ["е", "ｅ", "𝐞", "𝑒"],
        "f": ["ｆ", "𝐟", "𝑓", "𝒇"],
        "g": ["ɡ", "ｇ", "𝐠", "𝑔"],
        "h": ["һ", "ｈ", "𝐡", "𝒉"],
        "i": ["і", "ｉ", "𝐢", "𝑖"],
        "j": ["ϳ", "ｊ", "𝐣", "𝑗"],
        "k": ["ᴋ", "ｋ", "𝐤", "𝑘"],
        "l": ["І", "ｌ", "𝐥", "𝑙"],
        "m": ["ｍ", "𝐦", "𝑚", "𝒎"],
        "n": ["ո", "ｎ", "𝐧", "𝑛"],
        "o": ["о", "ｏ", "𝐨", "𝑜"],
        "p": ["р", "ｐ", "𝐩", "𝑝"],
        "q": ["ԛ", "ｑ", "𝐪", "𝑞"],
        "r": ["г", "ｒ", "𝐫", "𝑟"],
        "s": ["ѕ", "ｓ", "𝐬", "𝑠"],
        "t": ["т", "ｔ", "𝐭", "𝑡"],
        "u": ["υ", "ｕ", "𝐮", "𝑢"],
        "v": ["ν", "ｖ", "𝐯", "𝑣"],
        "w": ["ԝ", "ｗ", "𝐰", "𝑤"],
        "x": ["х", "ｘ", "𝐱", "𝑥"],
        "y": ["у", "ｙ", "𝐲", "𝑦"],
        "z": ["ᴢ", "ｚ", "𝐳", "𝑧"],
        "A": ["А", "Ａ", "𝐀", "𝐴"],
        "B": ["В", "Ｂ", "𝐁", "𝐵"],
        "C": ["С", "Ｃ", "𝐂", "𝐶"],
        "D": ["Ꭰ", "Ｄ", "𝐃", "𝐷"],
        "E": ["Е", "Ｅ", "𝐄", "𝐸"],
        "F": ["Ϝ", "Ｆ", "𝐅", "𝐹"],
        "G": ["Ԍ", "Ｇ", "𝐆", "𝐺"],
        "H": ["Н", "Ｈ", "𝐇", "𝐻"],
        "I": ["І", "Ｉ", "𝐈", "𝐼"],
        "J": ["Ј", "Ｊ", "𝐉", "𝐽"],
        "K": ["К", "Ｋ", "𝐊", "𝐾"],
        "L": ["Ⅼ", "Ｌ", "𝐋", "𝐿"],
        "M": ["М", "Ｍ", "𝐌", "𝑀"],
        "N": ["Ν", "Ｎ", "𝐍", "𝑁"],
        "O": ["О", "Ｏ", "𝐎", "𝑂"],
        "P": ["Р", "Ｐ", "𝐏", "𝑃"],
        "Q": ["Ԛ", "Ｑ", "𝐐", "𝑄"],
        "R": ["Ꭱ", "Ｒ", "𝐑", "𝑅"],
        "S": ["Ѕ", "Ｓ", "𝐒", "𝑆"],
        "T": ["Т", "Ｔ", "𝐓", "𝑇"],
        "U": ["Ս", "Ｕ", "𝐔", "𝑈"],
        "V": ["Ѵ", "Ｖ", "𝐕", "𝑉"],
        "W": ["Ԝ", "Ｗ", "𝐖", "𝑊"],
        "X": ["Х", "Ｘ", "𝐗", "𝑋"],
        "Y": ["Υ", "Ｙ", "𝐘", "𝑌"],
        "Z": ["Ꮓ", "Ｚ", "𝐙", "𝑍"],
    }

    # Variation selectors
    VARIATION_SELECTORS = [chr(0xFE00 + i) for i in range(16)]

    # Combining chracters that don't substantially change appearance
    COMBINING_CHARS = [
        "\u0305",  # Combining overline
        "\u0311",  # Combining inverted breve
        "\u0316",  # Combining grave accent below
        "\u0317",  # Combining acute accent below
        "\u032d",  # Combining circumflex accent below
        "\u0331",  # Combining macron below
        "\u0310",  # Combining candrabindu
    ]

    def __init__(self, secret_key: str = None, security_level: int = 1):
        """
        Initialize the TextWatermark class.

        Args:
        -----------
        secret_key: (str) Optional secret key for encryption/decryption.
        security_level: (int) from 1-3 determining watermark strength/visibility tradeoff
                1 = Highest invisibility, lower security
                2 = Balanced approach (default)
                3 = Highest security, potential minor visibility
        """
        self.security_level = min(max(1, security_level), 3)
        if secret_key is None:
            random.seed(42)
            self.secret_key = hashlib.sha256(
                str(random.getrandbits(256)).encode()
            ).hexdigest()
        else:
            self.secret_key = hashlib.sha256(secret_key.encode()).hexdigest()

        # Generate a stable random generator from the key
        self.random = random.Random(self.secret_key)

        # Set embedding density based on security level
        self.embedding_rates = {
            1: 0.05,  # Embed in 5% of possible locations
            2: 0.10,  # Embed in 10% of possible locations
            3: 0.20,  # Embed in 20% of possible locations
        }

        # Confdigure which techniques to use based on security level
        self.techniques = {
            1: ["zero_width"],
            2: ["zero_width", "homoglyphs"],
            3: ["zero_width", "homoglyphs", "variation"],
        }

    def _text_to_bit_array(self, text: str) -> List[int]:
        """
        Convert a string to an array of bits
        """
        bytes_data = text.encode("utf-8")
        binary = bin(int.from_bytes(bytes_data, "big"))[2:]

        # Ensure the binary string length is a multiple of 8
        padding = 8 - (len(binary) % 8) if len(binary) % 8 != 0 else 0
        binary = "0" * padding + binary

        # Convert binary string to list of bits
        bit_array = [int(bit) for bit in binary]
        return bit_array

    def _bit_array_to_text(self, bit_array: List[int]) -> str:
        """
        Convert an array of bits to a string
        """

        binary_str = "".join(str(bit) for bit in bit_array)

        try:
            n = int(binary_str, 2)
            bytes_data = n.to_bytes((n.bit_length() + 7) // 8, byteorder="big")
            return bytes_data.decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            return ""

    def _calculate_checksum(self, watermark: str) -> str:
        """
        Calculate a checksum for the watermark using SHA-256 to verify integrity.
        """
        return hashlib.md5(watermark.encode()).hexdigest()[:8]  # 8-char checksum

    def _get_embedding_positions(self, text: str, watermark_length: int) -> List[int]:
        """
        Determien positions in the text where the watermark bits will be embedded.
        Use a key-based selection to ensure the same positions are chosen during extraction.
        """
        # Get all possible positions (btn characters or word boundaries)
        text_length = len(text)
        word_boundaries = [m.start() for m in re.finditer(r"\b", text)]

        # Add character positions based on security level
        char_positions = []
        if self.security_level >= 2:
            # Add after every alphanumeric character
            char_positions = [
                i + 1
                for i in range(text_length - 1)
                if text[i].isalnum() and not text[i + 1].isspace()
            ]

        # Combine all potential positions and remove duplicates
        all_positions = sorted(set(word_boundaries + char_positions))

        # If too few positions, add more btn characters
        if (
            len(all_positions) < watermark_length * 3
        ):  # Need multiple options for each bit
            char_positions = [
                i
                for i in range(1, text_length)
                if not text[i].isspace() or not text[i - 1].isspace()
            ]
            all_positions = sorted(set(all_positions + char_positions))

        # Use secret key to determnistically select positions
        # Create a stable shuffling of positions based on the key
        seeded_positions = list(all_positions)
        self.random.shuffle(seeded_positions)

        # Calculate how many psitions to use based on security level
        num_positions = min(
            int(len(seeded_positions) * self.embedding_rates[self.security_level]),
            watermark_length * 3,  # Ensure we have 3x redundancy
        )

        # Return selected positions
        return seeded_positions[: max(num_positions, watermark_length)]

    def _select_embedding_technique(self, position: int, text: str) -> str:
        """
        Select the embedding technique based on the position and security level.
        """
        available = self.techniques[self.security_level]

        # Hash the position with the key to get a deterministic but seemingly random choice
        hash_input = f"{self.secret_key}:{position}:{text[position:position+2]}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)

        # Select a technique based on the hash
        return available[hash_value % len(available)]

    def _apply_zero_width_embedding(self, position: int, text: str, bit: int) -> str:
        """
        Embed a bit using zero-width characters.

        Args:
        -------
        position: (int) Position in the text to embed the bit.
        text: (str) The original text.
        bit: (int) The bit to embed (0 or 1).

        Returns:
        -------
        str: The text with the embedded bit.
        """
        if bit == 0:
            # For bit 0, use ZWSP or ZWNJ
            char_options = [self.ZERO_WIDTH_CHARS[0], self.ZERO_WIDTH_CHARS[1]]
        else:
            # For bit 1, use ZWJ or word joiner
            char_options = [self.ZERO_WIDTH_CHARS[2], self.ZERO_WIDTH_CHARS[3]]

        # Select a specific character deterministically
        hash_input = f"{self.secret_key}:{position}:{text[position:position+1]}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        selected_char = char_options[hash_value % len(char_options)]

        # Insert the zero-width character at that particular position
        return text[:position] + selected_char + text[position:]

    def _apply_homoglyph_embedding(self, position: int, text: str, bit: int) -> str:
        """
        Embed a bit by replacing a character with a homoglyph.
        """
        # Try to find a character at or after this position that has homoglyphs
        for i in range(position, min(position + 5, len(text))):
            if text[i].lower() in self.HOMOGLYPHS:
                char = text[i].lower()
                alternatives = self.HOMOGLYPHS[char]

                # Select homoglyph based on bit value and deterministic selection
                # Use first half of homoglyphs for bit 0, second half for bit 1
                mid_point = len(alternatives) // 2
                if mid_point == 0:
                    continue

                if bit == 0:
                    options = alternatives[:mid_point]
                else:
                    options = alternatives[mid_point:]

                hash_input = f"{self.secret_key}:{position}:{text[i-1:i+2]}"
                hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
                replacement = options[hash_value % len(options)]

                return text[:i] + replacement + text[i + 1 :]

        # If no suitable character found, fall back to zero-width embedding
        return self._apply_zero_width_embedding(position, text, bit)

    def _apply_combining_embedding(self, position: int, text: str, bit: int) -> str:
        """
        Embed bit using combining characters
        """
        # Try to a find a letter at or after this position to add combining character
        for i in range(position, min(position + 5, len(text))):
            if text[i].isalnum():
                # Choose combining character based on bit value
                if bit == 0:
                    options = self.COMBINING_CHARS[: len(self.COMBINING_CHARS) // 2]
                else:
                    options = self.COMBINING_CHARS[len(self.COMBINING_CHARS) // 2 :]

                hash_input = f"{self.secret_key}:{position}:{text[i]}"
                hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
                selected_char = options[hash_value % len(options)]
                # COME BACK TO THIS
                return text[:i] + selected_char + text[i + 1 :]

        # If no suitable character found, fall back to zero-width embedding
        return self._apply_zero_width_embedding(position, text, bit)

    def _apply_variation_embedding(self, position: int, text: str, bit: int) -> str:
        """
        Embed bit using variation selectors
        """
        # Try to a find a letter at or after this position to add combining character
        for i in range(position, min(position + 5, len(text))):
            if text[i].isalnum():
                # Choose combining character based on bit value
                if bit == 0:
                    options = self.VARIATION_SELECTORS[
                        : len(self.VARIATION_SELECTORS) // 2
                    ]
                else:
                    options = self.VARIATION_SELECTORS[
                        len(self.VARIATION_SELECTORS) // 2 :
                    ]

                hash_input = f"{self.secret_key}:{position}:{text[i]}"
                hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
                selected_char = options[hash_value % len(options)]
                return text[:i] + selected_char + text[i + 1 :]

        # If no suitable character found, fall back to zero-width embedding
        return self._apply_zero_width_embedding(position, text, bit)

    def embed_watermark(self, text: str, watermark: str) -> str:
        """
        Embed the watermark

        Args:
        -------
        text: (str) The original text.
        watermark: (str) The watermark to embed.

        Returns:
        -------
        str: The watermarked text.
        """
        if not text or not watermark:
            return text

        # Add a chucksum to watermark for verification
        checksum = self._calculate_checksum(watermark)
        full_watermark = f"{watermark}:{checksum}"

        watermark_bits = self._text_to_bit_array(full_watermark)

        positions = self._get_embedding_positions(text, len(watermark_bits))

        if len(positions) < len(watermark_bits):
            raise ValueError("Text too short to embed the entire watermark securely")

        # Embed each bit of the watermark
        result = text
        bits_embedded = 0
        offsets = {}  # Track offests caused by inserting characters

        for i, position in enumerate(positions):
            if bits_embedded >= len(watermark_bits):
                break

            bit_index = i % len(watermark_bits)
            bit = watermark_bits[bit_index]

            # Adjust position based on previously added offests
            adjusted_position = position
            for pos in sorted(offsets.keys()):
                if pos <= position:
                    adjusted_position += offsets[pos]

            # Select embedding technique
            technique = self._select_embedding_technique(adjusted_position, result)

            length_before = len(result)
            if technique == "zero_width":
                result = self._apply_zero_width_embedding(
                    adjusted_position, result, bit
                )
            elif technique == "homoglyphs":
                result = self._apply_homoglyph_embedding(adjusted_position, result, bit)
            elif technique == "variation":
                result = self._apply_variation_embedding(adjusted_position, result, bit)
            elif technique == "combining":
                result = self._apply_combining_embedding(adjusted_position, result, bit)
            else:
                raise ValueError(f"Unknown embedding technique: {technique}")

            # Track the offest created by this embedding
            length_after = len(result)
            offset = length_after - length_before
            if offset > 0:
                offsets[position] = offset

            bits_embedded += 1
        return result

    def _detect_zero_width(
        self, text: str, positions: List[int]
    ) -> List[Tuple[int, int]]:
        """
        Detect zero-width characters in the text at specified positions.
        """
        results = []
        for pos in positions:
            if pos < len(text):
                # Look for zero-w chars at this position
                if text[pos] in self.ZERO_WIDTH_CHARS:
                    if text[pos] in [
                        self.ZERO_WIDTH_CHARS[0],
                        self.ZERO_WIDTH_CHARS[1],
                    ]:
                        results.append((pos, 0))  # bit 0
                    else:
                        results.append((pos, 1))  # bit 1
        return results

    def _detect_homoglyphs(
        self, text: str, positions: List[int]
    ) -> List[Tuple[int, int]]:
        """
        Detect homoglyphs in the text at specified positions.
        """
        pass

    def detect(self, text: str) -> str:
        """
        Extract the watermark from the text.
        """
        pass

    def verify(self, text: str, expected_watermark: str) -> bool:
        """
        Verify the integrity of the watermark in the text.

        Args:
        -------
        text: (str) The watermarked text.
        expected_watermark: (str) The expected watermark.

        Returns:
        -------
        bool: True if the watermark is valid, False otherwise.
        """
        detected = self.detect(text)
        return detected == expected_watermark

    def remove(self, text: str) -> str:
        """
        Remove the watermark from the text.
        """
        pass
